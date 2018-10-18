import sys
import os
import time
import logging
import argparse
import scipy
import numpy as np
import os.path as osp
from pprint import pprint
import tensorflow as tf
from tqdm import tqdm

from loader import pad_batch
from models import (im_nets, seg_nets, quat_nets, model_silhouette,
    load_pretrained_im_vgg16, model_quat)
from silhonet import SilhoNet
from ops import image_sum, segment_sum, segment_sum_test, quat_losses, loss_sigmoid_cross_entropy
from ycb import YCB
from utils import Timer, get_session_config, init_logging, mkdir_p, process_args, write_args
from evaluate import (eval_seq_iou, init_angle_errors, init_matches, init_iou, print_iou_stats,
    print_match_stats, update_iou, update_matches, update_angle_errors_quat, eval_quat_match)

MODE_LIST = ['train-seg', 'test-seg', 'train-quat', 'test-quat']


def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print([str(i.name) for i in not_initialized_vars]) # for debugging
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def train(sess, net, sum_writer, args):
    # Add loss to the graph
    if args.net_mode == 'seg':
        net.loss_segment = loss_sigmoid_cross_entropy(net.pred_seg_map, net.gt_segment)
        net.loss_occluded_segment = loss_sigmoid_cross_entropy(net.pred_occ_seg_map, net.gt_occluded_segment)
        net.loss_segment = net.loss_segment + net.loss_occluded_segment
        net.loss = net.loss_segment
    elif args.net_mode == 'quat':
        net.loss_pose = quat_losses[args.quat_loss](net.pred_quat, net.gt_quat, args.batch_size)
        net.loss = net.loss_pose

    _t_dbg = Timer()

    # Add optimizer
    decay_lr    = tf.train.exponential_decay(
        args.lr,
        net.global_step,
        args.decay_steps,
        args.decay_rate,
        staircase=True)
    optimizer = tf.train.AdamOptimizer(decay_lr)
    optim     = optimizer.minimize(net.loss, net.global_step)
    # optim = tf.train.AdamOptimizer(decay_lr).minimize(net.loss, net.global_step)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Compute gradients and add gradient summary
    # Note: some gradients will be skipped if stop_gradient() is used
    grads = optimizer.compute_gradients(net.loss)
    grad_sum_list = []
    for index, grad in enumerate(grads):
        try:
            grad_sum = tf.summary.histogram("{}-grad".format(grad[1].name), grad)
            grad_sum_list.append(grad_sum)
        except Exception as e:
            logging.error(repr(e))
            logger.error('Skipping gradient for layer: {}'.format(grad[1]))
            continue
    net.grad_sum = tf.summary.merge(grad_sum_list)

    # Add summaries for training
    scalar_summary_list = []
    summary_list = []
    net.loss_sum = tf.summary.scalar('loss_sum', net.loss)
    lr_sum = tf.summary.scalar('lr', decay_lr)
    scalar_summary_list.extend((net.loss_sum, lr_sum))
    net.im_sum = image_sum(net.mdl_ims, net.batch_size, net.mdl_im_batch)
    net.seg_sum = segment_sum(net, net.prob_seg_map, net.gt_segment, tag='')
    net.occ_seg_sum = segment_sum(net, net.prob_occ_seg_map, net.gt_occluded_segment, tag='occ_')
    summary_list.extend((net.im_sum, net.seg_sum, net.occ_seg_sum))
    if args.net_mode == 'seg':
        net.loss_segment_sum = tf.summary.scalar('loss_segment_sum', net.loss_segment)
        net.loss_occluded_segment_sum = tf.summary.scalar('loss_occluded_segment_sum', net.loss_occluded_segment)
        scalar_summary_list.extend((net.loss_segment_sum, net.loss_occluded_segment_sum))
    elif args.net_mode == 'quat':
        net.loss_pose_sum = tf.summary.scalar('loss_pose_sum', net.loss_pose)
        scalar_summary_list.append(net.loss_pose_sum)
    merged_scalars = tf.summary.merge(scalar_summary_list)
    merged = tf.summary.merge(summary_list)

    # Add summaries for validation
    if args.run_val:
        # use streaming mean for validation loss calculation
        net.loss_val, net.loss_val_update = tf.metrics.mean(net.loss, name='loss_val_mean')
        # Isolate the variables stored behind the scenes by the metric operation
        net.loss_val_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='loss_val_mean')
        net.loss_val_vars_init = tf.variables_initializer(var_list=net.loss_val_vars)
        net.loss_val_sum = tf.summary.scalar('loss_val', net.loss_val)
        if args.net_mode == 'seg':
            net.seg_val_sum = segment_sum(net, net.prob_seg_map, net.gt_segment, tag='val_')
            net.occ_seg_val_sum = segment_sum(net, net.prob_seg_map, net.gt_segment, tag='val_occ_')
            net.seg_val_sum = tf.summary.merge([net.occ_seg_val_sum, net.seg_val_sum])

    # Initialize dataset
    coord_train = tf.train.Coordinator()

    if args.run_val:
        coord_val = tf.train.Coordinator()
        dset_val = YCB(
            image_set='keyframe',
            roi_size=args.roi_size,
            roi_area_thresh=args.roi_area_thresh,
            roi_fill_thresh=args.roi_fill_thresh,
            shuffle=args.shuffle,
            rng_seed=args.rng_seed,
            mode='train',
            quat_regress=(args.net_mode == 'quat'))

        dset_val.init_queue(
            coord_val,
            qsize=args.prefetch_qsize,
            nthreads=args.prefetch_threads)

    dset_train = YCB(
        image_set='trainsyn',
        roi_size=args.roi_size,
        roi_area_thresh=args.roi_area_thresh,
        roi_fill_thresh=args.roi_fill_thresh,
        shuffle=args.shuffle,
        rng_seed=args.rng_seed,
        mode='train',
        quat_regress=(args.net_mode == 'quat'))

    dset_train.init_queue(
        coord_train,
        nepochs=args.nepochs,
        qsize=args.prefetch_qsize,
        nthreads=args.prefetch_threads)

    # Restore checkpoint
    if args.ckpt is not None:
        logger.info('Restoring from %s', args.ckpt)
        # saver.restore(sess, args.ckpt)
        optimistic_restore(sess, args.ckpt)
    else:
        initialize_uninitialized(sess)
        sess.run(net.global_step.assign(0))

    logger.info('Training with all classes')

    # Training loop
    iters = 0
    pbar = tqdm(desc='Training SilhoNet', total=args.niters)
    try:
        while True:
            iters += 1
            _t_dbg.tic()
            batch_data = dset_train.next_batch(net.batch_size)
            logging.debug('Data read time - %.3fs', _t_dbg.toc())
            if args.net_mode == 'seg':
                feed_dict = {
                    net.rois:       batch_data['roi'],
                    net.mdl_ims:    batch_data['im_mdl'],
                    net.class_id:   batch_data['class_id'],
                    net.input_ims:  batch_data['im'],
                    net.gt_segment: batch_data['segment'],
                    net.gt_occluded_segment: batch_data['occluded_segment'],
                    net.keep_prob:  args.keep_prob
                }
            elif args.net_mode == 'quat':
                feed_dict = {
                    net.gt_quat:    batch_data['quat'],
                    net.class_id:   batch_data['class_id'],
                    net.gt_segment: batch_data['segment'],
                    net.keep_prob:  args.keep_prob
                }
            _t_dbg.tic()
            if args.run_trace and (iters % args.sum_iters == 0 or
                                   iters == 1 or iters == args.niters):
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                step_, _, merged_scalars_ = sess.run(
                    [net.global_step, optim, merged_scalars],
                    feed_dict=feed_dict,
                    options=run_options,
                    run_metadata=run_metadata)
                sum_writer.add_run_metadata(run_metadata, 'step%d' % step_)
            else:
                step_, _, merged_scalars_ = sess.run(
                    [net.global_step, optim, merged_scalars],
                    feed_dict=feed_dict)

            logging.debug('Net time - %.3fs', _t_dbg.toc())

            sum_writer.add_summary(merged_scalars_, step_)
            if args.net_mode == 'seg':
                if (iters % args.sum_iters == 0 or iters == 1 or iters == args.niters):
                    image_sum_, step_ = sess.run(
                        [merged, net.global_step], feed_dict=feed_dict)
                    sum_writer.add_summary(image_sum_, step_)
                    if args.vis_gradients:
                        grad_sum_ = sess.run(fetches=net.grad_sum, feed_dict=feed_dict)
                        sum_writer.add_summary(grad_sum_, step_)
            if iters % args.ckpt_iters == 0 or iters == args.niters:
                save_f = saver.save(
                    sess,
                    osp.join(args.log_dir, 'silhonet'),
                    global_step=net.global_step)
                logger.info(' Model checkpoint - {:s} '.format(save_f))
            if args.run_val and (iters % args.val_iters == 0 or iters == args.niters):
                logger.info('Calculating validation loss')
                # zero validation loss metric variables
                sess.run(net.loss_val_vars_init)
                iters_val = 0
                while True:
                    print('{}/{}     '.format(iters_val+1, args.niters_val), end='\r')
                    sys.stdout.flush()
                    iters_val += 1
                    batch_data_val = dset_val.next_batch(net.batch_size)
                    if args.net_mode == 'seg':
                        feed_dict_val = {
                            net.rois:       batch_data_val['roi'],
                            net.mdl_ims:    batch_data_val['im_mdl'],
                            net.class_id:   batch_data_val['class_id'],
                            net.input_ims:  batch_data_val['im'],
                            net.gt_segment: batch_data_val['segment'],
                            net.gt_occluded_segment: batch_data_val['occluded_segment'],
                            net.keep_prob: 1.0
                        }
                    elif args.net_mode == 'quat':
                        feed_dict_val = {
                            net.gt_quat:    batch_data_val['quat'],
                            net.class_id:   batch_data_val['class_id'],
                            net.gt_segment: batch_data_val['segment'],
                            net.keep_prob:  args.keep_prob
                        }
                    sess.run([net.loss_val_update], feed_dict=feed_dict_val)
                    if iters_val % args.sum_iters == 0 or iters_val == 1 or iters_val == args.niters_val:
                        if args.net_mode == 'seg':
                            seg_val_sum_ = sess.run(fetches=net.seg_val_sum, feed_dict=feed_dict_val)
                            sum_writer.add_summary(seg_val_sum_, step_)
                    if iters_val >= args.niters_val:
                        break
                loss_val_sum_ = sess.run(net.loss_val_sum)
                sum_writer.add_summary(loss_val_sum_, step_)
            pbar.update(1)
            if iters >= args.niters:
                break
    except Exception as e:
        logging.error(repr(e))
        dset_train.close_queue(e)
        dset_val.close_queue(e)
    finally:
        pbar.close()
        logger.info('Training completed')
        dset_train.close_queue()
        coord_train.join()
        if args.run_val:
            dset_val.close_queue()
            coord_val.join()


def test_silhouette(sess, net, sum_writer, args):
    result_file = args.log_dir + "/silhouette_table.txt"
    coord = tf.train.Coordinator()
    increment_global_step_op = tf.assign(net.global_step, net.global_step+1)

    # Init IoU
    iou = init_iou(args.eval_thresh)
    iou_occ = init_iou(args.eval_thresh)

    net.seg_sum = segment_sum_test(net, net.prob_seg_map, net.gt_segment, args.eval_thresh)
    net.seg_occ_sum = segment_sum_test(net, net.prob_occ_seg_map, net.gt_occluded_segment, args.eval_thresh, tag='_occ')
    net.seg_sum = tf.summary.merge([net.seg_sum, net.seg_occ_sum])
    net.im_sum  = image_sum(net.mdl_ims, net.batch_size, net.mdl_im_batch)

    # Init dataset
    dset = YCB(
        image_set=args.split,
        roi_size=args.roi_size,
        roi_area_thresh=args.roi_area_thresh,
        roi_fill_thresh=args.roi_fill_thresh,
        shuffle=False,
        rng_seed=0,
        mode='test',
        use_pred_rois=args.use_pred_rois)

    dset.init_queue(
        coord,
        nepochs=1,
        qsize=args.prefetch_qsize,
        nthreads=args.prefetch_threads)

    im_ids = dset.get_im_ids()

    # Restore checkpoint
    logger.info('Restoring from %s', args.ckpt)
    # saver = tf.train.Saver()
    # saver.restore(sess, args.ckpt)
    optimistic_restore(sess, args.ckpt)
    sess.run(net.global_step.assign(0))

    logger.info('Testing with all models')

    # Testing loop
    pbar = tqdm(desc='Testing', total=len(im_ids))
    deq_mids, deq_sids = [], []
    im_set = []
    sum_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    iters = 0
    try:
        while not coord.should_stop():
            iters += 1
            batch_data = dset.next_batch(net.batch_size)
            if batch_data is None:
                continue
            deq_sids.append(batch_data['seq_id'])
            deq_mids.append(batch_data['model_id'])
            num_batch_items = batch_data['class_id'].shape[0]
            batch_data = pad_batch(batch_data, args.batch_size)
            feed_dict = {
                net.rois:         batch_data['roi'],
                net.mdl_ims:      batch_data['im_mdl'],
                net.class_id:     batch_data['class_id'],
                net.input_ims:    batch_data['im'],
                net.gt_segment:   batch_data['segment'],
                net.gt_occluded_segment: batch_data['occluded_segment'],
                net.keep_prob:    1.0
            }

            pred, pred_occ = sess.run([net.prob_seg_map, net.prob_occ_seg_map], feed_dict=feed_dict)

            # Visualize segmentations
            if args.use_summary and (iters == 1 or iters % args.sum_iters == 0):
                step_, seg_sum_, im_sum_ = sess.run([increment_global_step_op, net.seg_sum, net.im_sum], feed_dict=feed_dict)
                sum_writer.add_summary(seg_sum_, step_)
                sum_writer.add_summary(im_sum_, step_)
                sum_writer.flush()

            # Calculate IOUs
            batch_iou = eval_seq_iou(
                pred,
                batch_data['segment'],
                thresh=args.eval_thresh)
            batch_iou_occ = eval_seq_iou(
                pred_occ,
                batch_data['occluded_segment'],
                thresh=args.eval_thresh)

            # Update iou dict
            iou = update_iou(batch_iou, iou)
            iou_occ = update_iou(batch_iou_occ, iou_occ)
            
            # Iterate progress bar by number of new images processed
            itr_len_prev = len(im_set)
            im_set = set().union(im_set, batch_data['im_index'])
            itr_delta = len(im_set) - itr_len_prev
            pbar.update(itr_delta)
    except Exception as e:
        logger.error(repr(e))
        dset.close_queue(e)
    finally:
        pbar.close()
        sess.close()
        logger.info('Testing completed')
        coord.join()

    deq_mids = np.concatenate(deq_mids, axis=0)[..., 0].tolist()
    deq_sids = np.concatenate(deq_sids, axis=0)[..., 0].tolist()

    # Print statistics and save to file
    stats, iou_table = print_iou_stats(
        list(zip(deq_sids, deq_mids)), iou, args.eval_thresh, label='Unoccluded ')
    print(iou_table)
    stats_occ, iou_occ_table = print_iou_stats(
        list(zip(deq_sids, deq_mids)), iou_occ, args.eval_thresh, label='Occluded ')
    print(iou_occ_table)
    if result_file is not None:
        logger.info('Result written to: {:s}'.format(result_file))
        with open(result_file, 'w') as f:
            f.write(iou_table)
            f.write("\n")
            f.write(iou_occ_table)


def test_pose(sess, net, sum_writer, args):
    result_file = args.log_dir + "/pose_table.txt"
    result_mat_file = args.log_dir + "/angle_errors.mat"

    coord = tf.train.Coordinator()
    increment_global_step_op = tf.assign(net.global_step, net.global_step+1)

    net.seg_sum = segment_sum(net, net.prob_seg_map, net.gt_segment, tag='')
    net.occ_seg_sum = segment_sum(net, net.prob_occ_seg_map, net.gt_occluded_segment, tag='occ_')
    net.seg_sum = tf.summary.merge([net.seg_sum, net.occ_seg_sum])
    net.im_sum  = image_sum(net.mdl_ims, net.batch_size, net.mdl_im_batch)

    # Init dataset
    dset = YCB(
        image_set=args.split,
        roi_size=args.roi_size,
        roi_area_thresh=args.roi_area_thresh,
        roi_fill_thresh=args.roi_fill_thresh,
        shuffle=False,
        rng_seed=0,
        mode='test',
        use_pred_rois=args.use_pred_rois)

    dset.init_queue(
        coord,
        nepochs=1,
        qsize=args.prefetch_qsize,
        nthreads=args.prefetch_threads)

    im_ids = dset.get_im_ids()

    # Init embedding match array
    matches = init_matches(args.eval_thresh)
    angle_errors = init_angle_errors(dset._classes)

    # Restore checkpoint
    logger.info('Restoring from %s', args.ckpt)
    # saver.restore(sess, args.ckpt)
    optimistic_restore(sess, args.ckpt)
    sess.run(net.global_step.assign(0))

    logger.info('Testing with all models')

    # Testing loop
    pbar = tqdm(desc='Testing', total=len(im_ids))
    deq_mids, deq_sids = [], []
    im_set = []
    sum_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
    iters = 0
    try:
        while not coord.should_stop():
            iters += 1
            batch_data = dset.next_batch(net.batch_size)
            if batch_data is None:
                continue
            deq_sids.append(batch_data['seq_id'])
            deq_mids.append(batch_data['model_id'])
            num_batch_items = batch_data['class_id'].shape[0]
            batch_data = pad_batch(batch_data, args.batch_size)
            feed_dict = {
                net.rois:         batch_data['roi'],
                net.gt_quat:      batch_data['quat'],
                net.mdl_ims:      batch_data['im_mdl'],
                net.class_id:     batch_data['class_id'],
                net.input_ims:    batch_data['im'],
                net.gt_segment:   batch_data['segment'],
                net.gt_occluded_segment: batch_data['occluded_segment'],
                net.keep_prob:    1.0
            }
            pred_quat = sess.run(net.pred_quat, feed_dict=feed_dict)
            # Calculate matches
            batch_quat_match = eval_quat_match(
                pred_quat,
                batch_data['quat'],
                thresh=args.eval_thresh)
            matches = update_matches(batch_quat_match, matches)
            # Calculate angle errors
            angle_errors = update_angle_errors_quat(
                angle_errors,
                dset._classes,
                pred_quat,
                batch_data['quat'],
                batch_data['class_id'])
            # Visualize segmentations
            if args.use_summary and (iters == 1 or iters % args.sum_iters == 0):
                step_, seg_sum_, im_sum_ = sess.run([increment_global_step_op, net.seg_sum, net.im_sum], feed_dict=feed_dict)
                sum_writer.add_summary(seg_sum_, step_)
                sum_writer.add_summary(im_sum_, step_)
                sum_writer.flush()
            # Iterate progress bar by number of new images processed
            itr_len_prev = len(im_set)
            im_set = set().union(im_set, batch_data['im_index'])
            itr_delta = len(im_set) - itr_len_prev
            pbar.update(itr_delta)
    except Exception as e:
        logger.error(repr(e))
        dset.close_queue(e)
    finally:
        pbar.close()
        sess.close()
        logger.info('Testing completed')
        coord.join()

    deq_mids = np.concatenate(deq_mids, axis=0)[..., 0].tolist()
    deq_sids = np.concatenate(deq_sids, axis=0)[..., 0].tolist()

    # Print statistics and save to file
    stats, match_table = print_match_stats(
        list(zip(deq_sids, deq_mids)), matches, args.eval_thresh)
    print(match_table)
    if result_file is not None:
        logger.info('Result written to: {:s}'.format(result_file))
        with open(result_file, 'w') as f:
            f.write(match_table)
    if result_mat_file is not None:
        logger.info('Result written to: {:s}'.format(result_mat_file))
        for key in angle_errors: angle_errors[key] = np.array(angle_errors[key], dtype=np.object)
        scipy.io.savemat(result_mat_file, angle_errors)


def parse_args():
    # Parameters for both testing and training
    parser = argparse.ArgumentParser(description='Options for SilhoNet')
    parser.add_argument('--mode', required=True, type=str, choices=MODE_LIST, help='mode for running network')
    parser.add_argument('--argsjs', type=str, default=None, help='path to json args file')
    parser.add_argument('--logdir', type=str, default='./log', help='logging directory. Automatically appends ' +
        '$DATE/train/ to path for training when not running from a checkpoint')
    parser.add_argument('--loglevel', type=str, default='info', help='logging level: error, info, warn, debug')
    parser.add_argument('--batch_size', type=int, default=4, help='input image batch size')
    parser.add_argument('--mdl_im_batch', type=int, default=12, help='number of rendered model images per ROI detection for silhouette prediction')
    parser.add_argument('--prefetch_threads', type=int, default=2, help='number of threads for processing dataset into queue')
    parser.add_argument('--prefetch_qsize', type=int, default=32, help='maximum size of input queue')
    parser.add_argument('--mdl_im_h', type=int, default=224, help='height of model input image')
    parser.add_argument('--mdl_im_w', type=int, default=224, help='width of model input image')
    parser.add_argument('--im_h', type=int, default=480, help='height of input image')
    parser.add_argument('--im_w', type=int, default=640, help='width of input image')
    parser.add_argument('--roi_size', type=int, default=64, help='size of output silhouette masks. Also size of resized ROIs')
    parser.add_argument('--roi_area_thresh', type=float, default=10, help='minimum area of detected ROI')
    parser.add_argument('--roi_fill_thresh', type=float, default=0.01, help='minimum percentage of ROI area where detected object is visible')
    parser.add_argument('--norm', type=str, default='IN', help='layer normalization: Instance Norm: IN, Batch Norm: BN., or None')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint')
    parser.add_argument('--seg_ckpt', type=str, default=None, help='heckpoint for silhouette prediction stage only')
    parser.add_argument('--gpus', type=str, default="0", help='GPUs visible to Tensorflow')
    parser.add_argument('--shuffle', action="store_true", help='shuffle dataset')
    parser.add_argument('--num_classes', type=int, default=22, help='number of classes in dataset including background class')
    parser.add_argument('--threshold_mask', action="store_false", help='threshold probability silhouette predictions into binary masks')
    parser.add_argument(
        '--im_net', type=str, default='vgg16', choices=im_nets.keys(), help='feature extraction network')
    parser.add_argument(
        '--seg_net', type=str, default='stack', choices=seg_nets.keys(), help='silhouette prediction network')
    parser.add_argument(
        '--quat_net', type=str, default='quat_res', choices=quat_nets.keys(), help='3D pose regression network')
    parser.add_argument(
        '--quat_loss', type=str, default='log_dist', choices=quat_losses.keys(), help='oss function for 3D pose regression')
    # Parameters for training only
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--decay_rate', type=float, default=1, help='learning rate decay rate')
    parser.add_argument('--decay_steps', type=int, default=10000, help='period to apply learning rate decay')
    parser.add_argument('--niters', type=int, default=10000, help='iterations to train')
    parser.add_argument('--niters_val', type=int, default=1000, help='period of iterations to run validation')
    parser.add_argument('--sum_iters', type=int, default=50, help='period of iterations to run summery')
    parser.add_argument('--ckpt_iters', type=int, default=5000, help='period of iterations to save checkpoint')
    parser.add_argument('--val_iters', type=int, default=100, help='iterations over which to calculate validation loss')
    parser.add_argument('--rng_seed', type=int, default=0, help='random seed')
    parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train. If None, network trains for niters. If not None, netowrk trains ' +
        'for which comes first, niters or nepochs')
    parser.add_argument('--keep_prob', type=float, default=0.5, help='keep probability for dropout')
    parser.add_argument('--run_val', action="store_true", help='run validation during training')
    parser.add_argument('--vis_gradients', action="store_true", help='visualize layer gradients during training')
    parser.add_argument('--run_trace', action="store_true", help='run trace on network termination')
    parser.add_argument('--use_pretrained', action="store_true", help='use pretrained imagenet weights for initializing VGG16 backbone network')
    # Parameters for testing only
    parser.add_argument('--eval_thresh', type=float, nargs='+', help='list of threshold values at which to evaluate test performance. For silhouette ' +
        'prediction, values are used to threshold probability masks into binary masks. For 3D pose prediction, threshold values are maximum angle errors in degrees.')
    parser.add_argument('--split', type=str, default='keyframe', help='image set to evaluate.')
    args = process_args(parser)
    return args


def define_graph(sess, args):
    net = SilhoNet(
        im_bs=args.batch_size,
        mdl_im_bs=args.mdl_im_batch,
        im_h=args.im_h,
        im_w=args.im_w,
        mdl_im_h=args.mdl_im_h,
        mdl_im_w=args.mdl_im_w,
        roi_size=args.roi_size,
        num_classes=args.num_classes,
        mode=args.run_mode,
        norm=args.norm)

    net.gt_occluded_segment = tf.placeholder(tf.float32, net.roi_shape, name='gt_occluded_segment')
    net.gt_segment = tf.placeholder(tf.float32, net.roi_shape, name='gt_segment')
    net.input_ims  = tf.placeholder(tf.float32, net.input_im_tensor_shape, name='input_ims')
    net.keep_prob  = tf.placeholder(tf.float32, shape=(), name='keep_prob')
    net.class_id   = tf.placeholder(tf.int32, net.class_shape, name='class_ids')
    net.mdl_ims    = tf.placeholder(tf.float32, net.mdl_im_tensor_shape, name='mdl_ims')
    net.gt_quat    = tf.placeholder(tf.float32, net.gt_quat_shape, name='gt_quat')
    net.rois       = tf.placeholder(tf.float32, net.roi_tensor_shape, name='rois')
    net.global_step = tf.Variable(0, trainable=False, name='global_step')

    # Define silhouette prediction graph
    net = model_silhouette(
        net,
        im_nets[args.im_net],
        seg_nets[args.seg_net],
        args.threshold_mask)

    # Define pose graph
    if args.net_mode == 'quat':
        if args.seg_ckpt is not None:
            logger.info('Restoring segmentation net from %s', args.seg_ckpt)
            saver = tf.train.Saver()
            saver.restore(sess, args.seg_ckpt)
        net = model_quat(net, quat_nets[args.quat_net])

    if args.ckpt is None and args.seg_ckpt is None and args.use_pretrained:
        if args.im_net == 'vgg16':
            logger.info('Restoring pretrained vgg16 net')
            net = load_pretrained_im_vgg16(net, sess)

    return net


if __name__ == '__main__':
    key = time.strftime("%Y-%m-%d_%H%M%S")
    args = parse_args()
    init_logging(args.loglevel)
    logger = logging.getLogger('silhonet.' + __name__)
    logger.setLevel(logging.DEBUG)

    # Set visible GPUs for training
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpus

    # Create session
    sess = tf.Session(config=get_session_config())

    if 'train' in args.mode:
        args.run_mode = 'train'
    elif 'test' in args.mode:
        args.run_mode = 'test'
    if 'seg' in args.mode:
        args.net_mode = 'seg'
    elif 'quat' in args.mode:
        args.net_mode = 'quat'

    if args.run_mode == 'train':
        if args.ckpt is None:
            args.log_dir = osp.join(args.logdir, key, 'train')
        else:
            args.log_dir = args.logdir
    elif args.run_mode == 'test':
        if args.logdir is not None:
            args.log_dir = args.logdir
        else:
            args.log_dir = args.ckpt + "/test/"
        os.makedirs(os.path.dirname(args.log_dir), exist_ok=True)

    # Initialize network parameters
    net = define_graph(sess, args)

    # Set up logging
    mkdir_p(args.log_dir)
    write_args(args, osp.join(args.log_dir, 'args.json'))
    logger.info('Logging to {:s}'.format(args.log_dir))
    logger.info('\nUsing args:')
    pprint(vars(args))
    net.print_net()

    sum_writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    # Run loop
    if args.run_mode == 'train':
        train(sess, net, sum_writer, args)
    elif args.run_mode == 'test':
        if args.net_mode == 'seg':
            test_silhouette(sess, net, sum_writer, args)
        elif args.net_mode == 'quat':
            test_pose(sess, net, sum_writer, args)
