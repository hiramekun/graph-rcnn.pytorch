import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from structures.bounding_box_pair import BoxPairList
from torch.utils.data import DataLoader
from utils.comm import synchronize, get_rank
from utils.logger import setup_logger
from utils.miscellaneous import save_config
from utils.visualize import select_top_predictions, overlay_boxes, overlay_class_names

from lib.config import cfg
from lib.data.collate_batch import BatchCollator
from lib.data.transforms import build_transforms
from lib.model import SceneGraphGeneration
from lib.model import build_model
from my_evaluation import evaluate
from voc2019 import VOCDetection

label_human = ["man"]
label_target = ["helmet", "glove", "hat"]
num_relation = 50
th = -1


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(
        description="Graph Reasoning Machine for Visual Question Answering")
    parser.add_argument('--model', dest='model', help='options: grcnn, imp, msdn, nmotif',
                        default='grcnn', type=str)
    parser.add_argument('--backbone', dest='backbone', help='options: vgg16, res50, res101, res152',
                        default='vgg16', type=str)
    parser.add_argument('--dataset', dest='dataset', help='training dataset', default='vg_bm',
                        type=str)
    parser.add_argument('--start_epoch', dest='start_epoch', help='starting epoch', default=1,
                        type=int)
    parser.add_argument('--epochs', dest='max_epochs', help='number of epochs to train', default=20,
                        type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display', default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display', default=10000, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory to save models',
                        default="server", nargs=argparse.REMAINDER)
    parser.add_argument('--nw', dest='nworker', help='number of workers', default=0, type=int)
    parser.add_argument('--cuda', dest='cuda', help='whether use cuda', action='store_true')
    parser.add_argument('--bs', dest='batch_size', help='batch_size', default=1, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs', help='whether use multiple gpus for training',
                        action='store_true')
    parser.add_argument('--pretrain', dest='pretrain',
                        help='whether it is pretraining faster r-cnn', action='store_true')
    # config optimization
    parser.add_argument('--o', dest='optimizer', help='training optimizer', default="sgd", type=str)
    parser.add_argument('--lr', dest='lr_base', help='base learning rate', default=0.01, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch', default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session', help='training session', default=1, type=int)

    # training mode
    parser.add_argument('--mode', dest='mode',
                        help='training mode, 0:scratch, 1:resume or 2:finetune', default=0,
                        type=int)
    parser.add_argument('--checksession', dest='checksession', help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint', help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard', default=False, type=bool)

    parser.add_argument("--config-file", default="configs/baseline_res101.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--instance", type=int, default=-1)
    parser.add_argument("--use_freq_prior", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--algorithm", type=str, default='sg_baseline')
    parser.add_argument("--image", type=str)
    parser.add_argument("--dir", type=str)
    args = parser.parse_args()
    return args


def imshow(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_detection(dataset, img_id, imgs, prediction):
    visualize_folder = "my_visualize"
    if not os.path.exists(visualize_folder):
        os.mkdir(visualize_folder)
    top_prediction = select_top_predictions(prediction)
    # c, w, h -> w, h, c
    img = imgs.tensors[0].permute(1, 2, 0).contiguous().cpu().numpy() + np.array(
        cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
    result = img.copy()
    result = overlay_boxes(result, top_prediction)
    result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
    cv2.imwrite(os.path.join(visualize_folder, "detection_{}.jpg".format(img_id)),
                result)


def visualize_detection_from_raw(img_id, img, prediction):
    visualize_folder = "./my_visualize_holding_helmet"
    if not os.path.exists(visualize_folder):
        os.mkdir(visualize_folder)
    # top_prediction = select_top_predictions(prediction)
    top_prediction = prediction
    result = img.permute(1, 2, 0).contiguous().cpu().numpy().copy() + np.array(
        cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
    result = overlay_boxes(result, top_prediction)
    result = overlay_class_names(result, top_prediction,
                                 VOCDetection("./data", transforms=None).ind_to_classes)
    cv2.imwrite(os.path.join(visualize_folder, "detection_{}.jpg".format(img_id)),
                result)


def predict_imgs(dir_path, model):
    imgs = os.listdir(dir_path)
    if th > 0:
        tmp = th
    else:
        tmp = num_relation
    with open(f'result-holding-helmet-{tmp}.txt', 'w') as f:
        for img in imgs:
            predict_img(os.path.join(dir_path, img), model, f)


def predict_img(img_path, model, res_file=None):
    import time
    img = Image.open(img_path)
    img = np.asarray(img)
    print(img.shape)
    start = time.time()
    transforms = build_transforms(cfg, is_train=False)
    im = Image.fromarray(img)
    im, _ = transforms(im, None)

    with torch.no_grad():
        im = im.to(torch.device("cuda"))
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print("inference on " + img_name + "...")

        output = model.scene_parser(im)
        output, output_pred = output

        cpu_device = torch.device("cpu")

        output_pred: [BoxPairList] = [o.to(cpu_device) for o in output_pred]
        output = [o.to(cpu_device) for o in output]

        output = output[0]
        output_pred = output_pred[0]
        triplets, pred_box = evaluate(output.bbox, output.get_field("scores"),
                                      output.get_field("labels"),
                                      output_pred.get_field("idx_pairs"),
                                      output_pred.get_field("scores"), num_relation, th)
        visualize_detection_from_raw(img_name, im, output)
        write_str = img_name + ", "
        for (s, r, o) in triplets:
            s_str = itola[str(s)]
            o_str = itola[str(o)]
            r_str = itopred[str(r)]
            if s_str in label_human and o_str in label_target:
                sent = s_str + " " + r_str + " " + o_str
                write_str += sent + ", "

        print(write_str)
        if res_file:
            res_file.write(write_str + "\n")

    end = time.time()
    print("elapsed_time: {0} sec".format(end - start))


def predict_with_loader(model):
    import time
    start = time.time()
    transforms = build_transforms(cfg, is_train=False)
    dataset = VOCDetection("./data", transforms=transforms)

    collator = BatchCollator(cfg.DATASET.SIZE_DIVISIBILITY)
    dataloader = DataLoader(dataset, collate_fn=collator)
    if th > 0:
        tmp = th
    else:
        tmp = num_relation

    with open(f'result-{tmp}-hat.txt', "w") as f:
        for i_batch, data in enumerate(dataloader):
            with torch.no_grad():
                imgs, targets, img_names = data
                imgs = imgs.to(torch.device("cuda"))
                _img_name = img_names[0]
                img_name = os.path.splitext(os.path.basename(_img_name))[0]
                print("inference on " + img_name + "...")

                output = model.scene_parser(imgs)
                output, output_pred = output

                cpu_device = torch.device("cpu")

                output_pred: [BoxPairList] = [o.to(cpu_device) for o in output_pred]
                output = [o.to(cpu_device) for o in output]

                output = output[0]
                output_pred = output_pred[0]
                triplets, pred_box = evaluate(output.bbox, output.get_field("scores"),
                                              output.get_field("labels"),
                                              output_pred.get_field("idx_pairs"),
                                              output_pred.get_field("scores"), num_relation, th)
                visualize_detection(dataloader.dataset, img_name, imgs, output)
                write_str = img_name + ", "
                for (s, r, o) in triplets:
                    s_str = itola[str(s)]
                    o_str = itola[str(o)]
                    r_str = itopred[str(r)]
                    if s_str in label_human and o_str in label_target:
                        sent = s_str + " " + r_str + " " + o_str
                        write_str += sent + ", "
                f.write(write_str + "\n")
    end = time.time()
    print("elapsed_time: {0} sec".format(end - start))


if __name__ == '__main__':
    args = parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.resume = args.resume
    cfg.inference = args.inference
    cfg.MODEL.USE_FREQ_PRIOR = args.use_freq_prior
    cfg.MODEL.ALGORITHM = args.algorithm

    if not os.path.exists("logs") and get_rank() == 0:
        os.mkdir("logs")
    logger = setup_logger("scene_graph_generation", "logs", get_rank())
    logger.info(args)
    logger.info("Loaded configuration file {}".format(args.config_file))
    output_config_path = os.path.join("logs", 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    save_config(cfg, output_config_path)

    arguments = {"iteration": 0}
    model = build_model(cfg, arguments, args.local_rank, args.distributed)
    model.scene_parser.eval()

    info = json.load(open(os.path.join(cfg.DATASET.PATH, "VG-SGG-dicts.json"), 'r'))
    itola = info['idx_to_label']
    itopred = info['idx_to_predicate']

    if args.image:
        predict_img(args.image, model)
    elif args.dir:
        predict_imgs(args.dir, model)
    else:
        predict_with_loader(model)
