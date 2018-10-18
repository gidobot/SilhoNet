import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from math import acos
import logging

logger = logging.getLogger('silhonet.' + __name__)

############################################
##### Utility functions for evaluation #####
############################################


def init_iou(thresh):
    iou = dict()
    for k in thresh:
        iou[k] = []
    return iou


def init_matches(thresh):
    matches = dict()
    for k in thresh:
        matches[k] = []
    return matches


def init_angle_errors(classes):
    errors = dict()
    for i, c in enumerate(classes):
        errors[c[4:]] = []
    return errors


def update_iou(batch_iou, iou):
    for th in iou.keys():
        iou[th].extend(batch_iou[th])
    return iou


def update_matches(batch_matches, matches):
    for th in matches.keys():
        matches[th].extend(batch_matches[th])
    return matches


def eval_seq_iou(pred, gt, thresh=[0.1]):
    bs = gt.shape[0]
    gt = gt.astype(np.bool)
    iu = dict()
    for k in thresh:
        iu[k] = []
    for th in thresh:
        for ix in range(bs):
            pred_t = (pred[ix] > th).astype(np.bool)
            i = np.sum(np.logical_and(pred_t, gt[ix]))
            u = np.sum(np.logical_or(pred_t, gt[ix]))
            thiou = float(i) / u
            iu[th].append(thiou)
    return iu


def eval_quat_match(pred_quat, gt_quat, thresh):
    bs = pred_quat.shape[0]
    match = dict()
    for k in thresh:
        match[k] = []
    for i in range(bs):
        angle_diff = np.degrees(2*acos(abs(np.dot(pred_quat[i],gt_quat[i]))))
        for th in thresh:
            match[th].append(int(angle_diff <= th))
    return match


def update_angle_errors_quat(vp_angle_errors, classes, pred_quat, gt_quat, class_id):
    bs = pred_quat.shape[0]
    for i in range(bs):
        angle_diff = np.degrees(2*acos(abs(np.dot(pred_quat[i],gt_quat[i]))))
        logger.debug('Angle error: {}'.format(angle_diff))
        # if class_id[i] == 1:
        #     print("pred_ind: {}, vp_id: {}".format(pred_ind, vp_id[i]))
        vp_angle_errors[classes[class_id[i]][4:]].append(angle_diff)
    return vp_angle_errors


def eval_seq_seg_ycb(pred, thresh=[0.1]):
    bs = pred.shape[0]
    segs = dict()
    for k in thresh:
        segs[k] = []
    for th in thresh:
        for ix in range(bs):
            segs[th].append((pred[ix] > th).astype(int))
        segs[th] = np.asarray(segs[th])
    return segs


def print_match_stats(mids, matches, thresh, statistic='mean'):
    ''' mids: [(seq_id, model_id), ...]
        matches: {'threshold': [bool, ...]}
        output: Matches: Model_ids - mean match percentage'''

    def pline(s):
        return '\n' + '*' * 5 + ' ' + s + ' ' + '*' * 5

    model_ids, counts = np.unique([m[1] for m in mids], return_counts=True)

    mmatch = dict()
    for th in thresh:
        mmatch[th] = dict()
        for mid in model_ids:
            mmatch[th][mid] = []

    th_sum = {}
    for th in sorted(thresh):
        th_sum[th] = 0
        for mx, m in enumerate(mids):
            mmatch[th][m[1]].append(matches[th][mx])

    full_table = []
    full_table.append(pline('Accuracy Statistic: '.format(statistic)))
    print_table = []
    for mid, count in zip(model_ids, counts):
        print_table.append([mid])
        for th in sorted(thresh):
            if statistic == 'mean':
                print_table[-1].append(
                    np.array(mmatch[th][mid]).mean() * 100)
            th_sum[th] += np.array(mmatch[th][mid]).mean() * 100
        print_table[-1].append(count)
    print_table.append(["mean"])
    for th in sorted(thresh):
        print_table[-1].append(th_sum[th] / len(model_ids))

    full_table.append(
        tabulate(print_table, headers=[""]+sorted(matches.keys())+["count"], floatfmt=".2f"))

    return mmatch, '\n'.join(full_table)


def print_iou_stats(mids, iou, thresh, statistic='mean', label=''):
    ''' mids: [(seq_id, model_id), ...]
        iou: {'threshold': iou}
        output: IoU Thresh: Model_ids - mean iou'''
    def pline(s):
        return '\n' + '*' * 5 + ' ' + s + ' ' + '*' * 5
    model_ids = np.unique([m[1] for m in mids])
    miou = dict()
    for th in thresh:
        miou[th] = dict()
        for mid in model_ids:
            miou[th][mid] = []
    th_sum = {}
    for th in sorted(thresh):
        th_sum[th] = 0
        for mx, m in enumerate(mids):
            miou[th][m[1]].append(iou[th][mx])
    full_table = []
    full_table.append(pline(label+'Silhouette Accuracy Statistic: '.format(statistic)))
    print_table = []
    for mid in model_ids:
        print_table.append([mid])
        for th in sorted(thresh):
            if statistic == 'mean':
                print_table[-1].append(
                    np.array(miou[th][mid]).mean() * 100)
            elif statistic == 'median':
                print_table[-1].append(
                    np.median(np.array(miou[th][mid])) * 100)
            th_sum[th] += np.array(miou[th][mid]).mean() * 100
    print_table.append(["mean"])
    for th in sorted(thresh):
        print_table[-1].append(th_sum[th] / len(model_ids))
    full_table.append(
        tabulate(print_table, headers=[""]+sorted(iou.keys()), floatfmt=".2f"))
    return miou, '\n'.join(full_table)


def vis_ims(ims, mask=None):
    if mask is not None:
        ims[np.logical_not(mask)] = None
    im_disp = np.reshape(ims, [-1] + list(ims.shape[2:]))
    im_d = np.concatenate([i for i in im_disp], axis=1)
    plt.imshow(np.uint8(im_d[..., 0] * 255))
    plt.axis('off')


def eval_l1_err(pred, gt, mask=None, vis=False):
    pred = pred[:, 0, ...]
    bs, im_batch = pred.shape[0], pred.shape[1]
    if mask is None:
        nanmask = (gt < np.max(gt))
    range_mask = np.logical_and(pred > 2.0 - np.sqrt(3) * 0.5,
                                pred < 2.0 + np.sqrt(3) * 0.5)
    mask = np.logical_and(nanmask, range_mask)

    if vis:
        plt.subplot(5, 1, 1)
        vis_ims(mask)
        plt.title("Eval Mask")
        plt.subplot(5, 1, 2)
        vis_ims(pred / 10.0, mask=mask)
        plt.title("Pred")
        plt.subplot(5, 1, 3)
        vis_ims(gt / 10.0, mask=nanmask)
        plt.title("Gt")
        plt.subplot(5, 1, 4)
        vis_ims(np.logical_xor(mask, nanmask))
        plt.title("Gt Mask - Mask")
        plt.subplot(5, 1, 5)
        vis_ims(np.abs(pred - gt) / 10.0, mask=mask)
        plt.title("Masked L1 error")
        plt.show()

    l1_err = np.abs(pred - gt)
    l1_err_masked = np.ma.array(l1_err, mask=np.logical_not(mask))
    batch_err = []
    for b in range(bs):
        tmp = np.zeros((im_batch, ))
        for imb in range(im_batch):
            tmp[imb] = np.ma.median(l1_err_masked[b, imb])
        batch_err.append(np.nanmean(tmp))
    return batch_err