import json
import os

import numpy as np
import torch
from utils.comm import synchronize

from lib.data.build import build_data_loader
from lib.config import cfg

from lib.data.evaluation.sg.evaluator import BasicSceneGraphEvaluator

PREFIX_RESULTS = "results/sg_imp_freq"


def evaluate(obj_rois, obj_scores, obj_labels,
             rel_inds, rel_scores,
             top_Ns, result_dict,
             mode, iou_thresh=0.5):
    obj_rois = obj_rois.cpu()
    obj_scores = obj_scores.cpu()
    obj_labels = obj_labels.cpu()
    rel_inds = rel_inds.cpu()
    rel_scores = rel_scores.cpu()

    # label = (((gt_rel_label.sum(1) == 0).int() + (gt_rel_label.sum(0) == 0).int()) == 2)
    # change_ix = label.nonzero()

    # pred
    box_preds = obj_rois.numpy()
    num_boxes = box_preds.shape[0]

    predicate_preds = rel_scores.numpy()

    # no bg
    predicate_preds = predicate_preds[:, 1:]
    predicates = np.argmax(predicate_preds, 1).ravel() + 1
    predicate_scores = predicate_preds.max(axis=1).ravel()

    relations = rel_inds.numpy()

    # if relations.shape[0] != num_boxes * (num_boxes - 1):
    # pdb.set_trace()

    # assert(relations.shape[0] == num_boxes * (num_boxes - 1))
    assert (predicates.shape[0] == relations.shape[0])
    # if scene graph detection task
    # use preicted boxes and predicted classes
    classes = obj_labels.numpy()  # np.argmax(class_preds, 1)
    class_scores = obj_scores.numpy()  # class_preds.max(axis=1)
    # boxes = []
    # for i, c in enumerate(classes):
    #     boxes.append(box_preds[i, c*4:(c+1)*4])
    # boxes = np.vstack(boxes)
    boxes = box_preds

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores, is_pred=False)
    sorted_inds = np.argsort(relation_scores)[::-1]
    # sorted_inds_obj = np.argsort(class_scores)[::-1]
    # # compue recall
    #
    # for k in result_dict[mode + '_recall']:
    #     this_k = min(k, num_relations)
    #     keep_inds = sorted_inds[:this_k]
    #     keep_inds_obj = sorted_inds_obj[:this_k]
    #
    #     # triplets_valid = _relation_recall_triplet(gt_triplets,
    #     #                           pred_triplets[keep_inds,:],
    #     #                           gt_triplet_boxes,
    #     #                           pred_triplet_boxes[keep_inds,:],
    #     #                           iou_thresh)
    #
    #     recall = _relation_recall(gt_triplets,
    #                               pred_triplets[keep_inds, :],
    #                               gt_triplet_boxes,
    #                               pred_triplet_boxes[keep_inds, :],
    #                               iou_thresh)
    #     num_gt = gt_triplets.shape[0]
    #
    #     result_dict[mode + '_recall'][k].append(recall / num_gt)
    #     # result_dict[mode + '_triplets'][k].append(triplets_valid)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores, is_pred=False):
    # format predictions into triplets

    # compute the overlaps between boxes
    if is_pred:
        overlaps = bbox_overlaps(torch.from_numpy(boxes).contiguous(),
                                 torch.from_numpy(boxes).contiguous())

    assert (predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    triplet_scores = np.zeros([num_relations]).astype(np.float32)
    for i in range(num_relations):
        triplets[i, 1] = predicates[i]
        sub_i, obj_i = relations[i, :2]
        triplets[i, 0] = classes[sub_i]
        triplets[i, 2] = classes[obj_i]
        triplet_boxes[i, :4] = boxes[sub_i, :]
        triplet_boxes[i, 4:] = boxes[obj_i, :]
        # compute triplet score
        score = class_scores[sub_i]
        score *= class_scores[obj_i]

        if is_pred:
            if overlaps[sub_i, obj_i] == 0:
                score *= 0
            else:
                score *= predicate_scores[i]
        else:
            score *= predicate_scores[i]

        triplet_scores[i] = score
    return triplets, triplet_boxes, triplet_scores


def setup_gpu():
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        print("use gpu...")
        torch.cuda.set_device(0)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
    return distributed


if __name__ == '__main__':
    predictions_pred = torch.load(f'{PREFIX_RESULTS}/predictions_pred.pth')
    predictions = torch.load(f'{PREFIX_RESULTS}/predictions.pth')

    data_loader_test = build_data_loader(cfg, split="test", is_distributed=setup_gpu())
    info = json.load(open(os.path.join(cfg.DATASET.PATH, "VG-SGG-dicts.json"), 'r'))
    itola = info['idx_to_label']
    print(itola)
    itopred = info['idx_to_predicate']
    print(itopred)

    dataset = data_loader_test.dataset
    for image_id, (prediction, prediction_pred) in enumerate(zip(predictions, predictions_pred)):
        # TODO: this is for instance
        if image_id > 10:
            break
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        print(f'image_width: {image_width}')
        print(f'image_height: {image_height}')

        gt_boxlist = dataset.get_groundtruth(image_id)

        gt_entry = {
            'gt_classes': gt_boxlist.get_field("labels").numpy(),
            'gt_relations': gt_boxlist.get_field("relation_labels").numpy().astype(int),
            'gt_boxes': gt_boxlist.bbox.numpy(),
        }

        # import pdb; pdb.set_trace()
        prediction = prediction.resize((image_width, image_height))
        obj_scores = prediction.get_field("scores").numpy()
        all_rels = prediction_pred.get_field("idx_pairs").numpy()
        fp_pred = prediction_pred.get_field("scores").numpy()
        # multiplier = np.ones((obj_scores.shape[0], obj_scores.shape[0]))
        # np.fill_diagonal(multiplier, 0)
        # fp_pred = fp_pred * multiplier.reshape(obj_scores.shape[0] * (obj_scores.shape[0] - 1), 1)
        scores = np.column_stack((
            obj_scores[all_rels[:, 0]],
            obj_scores[all_rels[:, 1]],
            fp_pred.max(1)
        )).prod(1)
        sorted_inds = np.argsort(-scores)
        sorted_inds = sorted_inds[scores[sorted_inds] > 0]  # [:100]

        pred_entry = {
            'pred_boxes': prediction.bbox.numpy(),
            'pred_classes': prediction.get_field("labels").numpy(),
            'obj_scores': prediction.get_field("scores").numpy(),
            'pred_rel_inds': all_rels[sorted_inds],
            'rel_scores': fp_pred[sorted_inds],
        }

        # output ground truth
        gt_classes = gt_entry["gt_classes"]
        gt_relations = gt_entry["gt_relations"]
        print("GROUND TRUTH")
        for (s, o, r) in gt_relations:
            # get class from detected objects
            s_class = gt_classes[s]
            o_class = gt_classes[o]

            # convert id to label string
            # key is string
            s_str = itola[str(s_class)]
            o_str = itola[str(o_class)]
            r_str = itopred[str(r)]

            print(f'{s_str} {r_str} {o_str}')
        print()

        # output prediction
        # got top 32
        triplets, _ = evaluate(prediction.bbox, prediction.get_field("scores"),
                               prediction.get_field("labels"),
                               prediction_pred.get_field("idx_pairs"),
                               prediction_pred.get_field("scores"),
                               20, {}, "")

        pred_classes = pred_entry["pred_classes"]
        obj_scores = pred_entry["obj_scores"]
        # more than 32. many relationships
        pred_rel_inds = pred_entry["pred_rel_inds"]
        rel_scores = pred_entry["rel_scores"]

        # o, s, r
        pred_rels = np.column_stack((pred_rel_inds, 1 + rel_scores[:, 1:].argmax(1)))

        print("PREDICTION")
        for (s, r, o) in triplets:
            s_str = itola[str(s)]
            o_str = itola[str(o)]
            r_str = itopred[str(r)]
            print(f'{s_str} {r_str} {o_str}')
        # for (so, rs) in zip(pred_rel_inds, rel_scores):
        #     s, o = so
        #     if obj_scores[s] < 0.5 or obj_scores[o] < 0.5:
        #         continue
        #     r = np.argmax(rs)
        #     s_class = pred_classes[s]
        #     o_class = pred_classes[o]
        #
        #     s_str = itola[str(s_class)]
        #     o_str = itola[str(o_class)]
        #     r_str = itopred[str(r)]
        #     print(f'{s_str} {r_str} {o_str}')
        print()

        print(pred_entry["pred_classes"].shape)
        print(pred_entry["pred_rel_inds"].shape)
        print(pred_entry["rel_scores"].shape)
