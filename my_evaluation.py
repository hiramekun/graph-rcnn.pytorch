import numpy as np


def evaluate(obj_rois, obj_scores, obj_labels, rel_inds, rel_scores, top_Ns, th=-1):
    obj_rois = obj_rois.cpu()
    obj_scores = obj_scores.cpu()
    obj_labels = obj_labels.cpu()
    rel_inds = rel_inds.cpu()
    rel_scores = rel_scores.cpu()

    box_preds = obj_rois.numpy()
    predicate_preds = rel_scores.numpy()
    predicate_preds = predicate_preds[:, 1:]
    predicates = np.argmax(predicate_preds, 1).ravel() + 1
    predicate_scores = predicate_preds.max(axis=1).ravel()

    relations = rel_inds.numpy()

    assert (predicates.shape[0] == relations.shape[0])

    classes = obj_labels.numpy()
    classes_scores = obj_scores.numpy()
    boxes = box_preds

    # relation scores means sub*obj*relation
    pred_triplets, pred_triplet_boxes, relation_scores = _triplet(predicates, relations, classes,
                                                                  boxes, predicate_scores,
                                                                  classes_scores)
    sorted_inds = np.argsort(relation_scores)[::-1]
    sorted_rel_scores = np.sort(relation_scores)[::-1]
    num_relations = relations.shape[0]
    if th > 0:
        bigger_than_th = sorted_rel_scores[sorted_rel_scores > th]
        k = min(len(bigger_than_th), num_relations)
    else:
        k = min(top_Ns, num_relations)
    print(k)
    keep_inds = sorted_inds[:k]

    return pred_triplets[keep_inds, :], pred_triplet_boxes[keep_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores):
    # format predictions into triplets

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
        score *= predicate_scores[i]

        triplet_scores[i] = score
    return triplets, triplet_boxes, triplet_scores
