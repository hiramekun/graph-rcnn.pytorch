import datetime
import json
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
from lib.scene_parser.rcnn.utils.comm import get_rank
from lib.scene_parser.rcnn.utils.logger import setup_logger
from utils.metric_logger import MetricLogger

from lib.config import cfg
from lib.data.build import build_data_loader
from lib.data.evaluation.sg.sg_eval import _triplet

label_human = ["man"]
label_target = ["helmet", "glove"]


def evaluate(gt_classes, gt_boxes, gt_rels):
    gt_classes = gt_classes.cpu()
    gt_boxes = gt_boxes.cpu()
    gt_rels = gt_rels.cpu()

    if gt_rels.ne(0).sum() == 0:
        return None, None

    rel_sum = ((gt_rels.sum(1) > 0).int() + (gt_rels.sum(0) > 0).int())

    # label = (((gt_rel_label.sum(1) == 0).int() + (gt_rel_label.sum(0) == 0).int()) == 2)
    # change_ix = label.nonzero()

    gt_boxes = gt_boxes.numpy()
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = gt_rels.nonzero().numpy()
    gt_classes = gt_classes.view(-1, 1).numpy()

    gt_rels_view = gt_rels.contiguous().view(-1)
    gt_pred_labels = gt_rels_view[gt_rels_view.nonzero().squeeze()].contiguous().view(-1, 1).numpy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return None, None
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_pred_labels,
                                                gt_relations,
                                                gt_classes,
                                                gt_boxes,
                                                gt_predicate_scores,
                                                gt_class_scores)
    return gt_triplets, gt_triplet_boxes


if __name__ == '__main__':
    info = json.load(open(os.path.join(cfg.DATASET.PATH, "VG-SGG-dicts.json"), 'r'))
    itola = info['idx_to_label']
    itopred = info['idx_to_predicate']
    meters = MetricLogger(delimiter="  ")
    data_loader = build_data_loader(cfg)
    end = time.time()
    logger = setup_logger("scene_graph_generation", "logs", get_rank())
    output_config_path = os.path.join("logs", 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))

    logger = logging.getLogger("scene_graph_generation")
    logger.info("Start training")
    max_iter = len(data_loader)
    result_dic: {str: int} = defaultdict(int)

    all_images = 0
    with open('browse_data.txt', 'w') as f:
        for i, data in enumerate(data_loader):
            data_time = time.time() - end

            imgs, target, idx = data
            all_images += len(imgs.tensors)
            gt_boxlist = data_loader.dataset.get_groundtruth(i)
            gt_triplets, _ = evaluate(gt_boxlist.get_field("labels"), gt_boxlist.bbox,
                                      gt_boxlist.get_field("pred_labels"))

            for (s, r, o) in gt_triplets:
                s_str = itola[str(s)]
                o_str = itola[str(o)]
                r_str = itopred[str(r)]
                if s_str in label_human and o_str in label_target:
                    sent = s_str + " " + r_str + " " + o_str
                    print(sent)
                    f.write(sent + '\n')
                    result_dic[sent] += 1

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if i % 20 == 0 or i == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "model: {tag}",
                            "eta: {eta}",
                            "iter: {iter}/{max_iter}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        tag="scene_parser",
                        eta=eta_string,
                        iter=i, max_iter=max_iter,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

    print(f'all_images, {all_images}')
    for sent, cnt in result_dic:
        print(f'{sent}, {cnt}')
