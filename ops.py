import tensorflow as tf
import tensorflow.contrib.slim as slim
import logging
import numpy as np
from math import acos, ceil

logger = logging.getLogger('silhonet.' + __name__)


def get_bias(shape, name='bias'):
    return tf.get_variable(
        name, shape=shape, initializer=tf.constant_initializer(0.0))


def get_weights(shape, name='weights'):
    return tf.get_variable(
        name, shape=shape, initializer=slim.initializers.xavier_initializer())


def instance_norm(x):
    epsilon = 1e-5
    x_shape = tf_static_shape(x)
    if len(x_shape) == 4:
        axis = [1, 2]
    elif len(x_shape) == 5:
        axis = [1, 2, 3]
    else:
        logger.error(
            'Instance norm not supported for tensor rank %d' % len(x_shape))
    with tf.variable_scope('InstanceNorm'):
        mean, var = tf.nn.moments(x, axis, keep_dims=True)
        beta = get_bias([x_shape[-1]])
        return tf.nn.batch_normalization(
            x, mean, var, offset=beta, scale=None, variance_epsilon=epsilon)


def deconv3d(name,
             X,
             fsize,
             ch,
             stride=2,
             norm=None,
             padding="SAME",
             activation=tf.nn.relu,
             mode="TRAIN"):
    bs, h, w, d, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, fsize, ch, in_ch]
    out_shape = [bs, h * stride, w * stride, d * stride, ch]
    stride = [1, stride, stride, stride, 1]
    with tf.variable_scope(name):
        if activation is not None:
            X = activation(X)

        params = get_weights(filt_shape)
        X = tf.nn.conv3d_transpose(X, params, out_shape, stride, padding)
        if norm is None:
            bias_dim = [filt_shape[-2]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def conv3d(name,
           X,
           fsize,
           ch,
           stride=2,
           norm=None,
           padding="SAME",
           activation=tf.nn.relu,
           mode="TRAIN"):

    bs, h, w, d, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, fsize, in_ch, ch]
    stride = [1, stride, stride, stride, 1]
    with tf.variable_scope(name):
        if activation is not None:
            X = activation(X)
        params = get_weights(filt_shape)
        X = tf.nn.conv3d(X, params, stride, padding)
        if norm is None:
            bias_dim = [filt_shape[-1]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def conv2d(name,
           X,
           fsize,
           ch,
           stride=1,
           norm=None,
           padding="SAME",
           act=tf.nn.relu,
           mode="TRAIN"):

    bs, h, w, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, in_ch, ch]
    stride = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if act is not None:
            X = act(X)

        params = get_weights(filt_shape)
        X = tf.nn.conv2d(X, params, stride, padding)

        if norm is None:
            bias_dim = [filt_shape[-1]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def make_deconv_filter(name, f_shape, trainable=True):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    var = tf.get_variable(name, shape=weights.shape, initializer=init, trainable=trainable)
    return var


def deconv_pcnn(input, k_h, k_w, c_o, s_h, s_w, name, reuse=None, padding='SAME', trainable=True):
        c_i = input.get_shape()[-1]
        with tf.variable_scope(name, reuse=reuse) as scope:
            # Compute shape out of input
            in_shape = tf_static_shape(input)
            h = in_shape[1] * s_h
            w = in_shape[2] * s_w
            new_shape = [in_shape[0], h, w, c_o]
            output_shape = tf.stack(new_shape)

            # filter
            f_shape = [k_h, k_w, c_o, c_i]
            weights = make_deconv_filter('weights', f_shape, trainable)
        return tf.nn.conv2d_transpose(input, weights, output_shape, [1, s_h, s_w, 1], padding=padding, name=scope.name)


def deconv2d(name,
             X,
             fsize,
             ch,
             stride=2,
             norm=None,
             padding="SAME",
             act=tf.nn.relu,
             mode="TRAIN"):

    bs, h, w, in_ch = tf_static_shape(X)
    filt_shape = [fsize, fsize, ch, in_ch]
    if padding == "SAME":
        out_shape = [bs, h * stride, w * stride, ch]
    elif padding == "VALID":
        out_shape = [bs, fsize + stride*(h-1), fsize + stride*(w-1), ch]
    else:
        logger.error('Invalid padding {}! Choose from {VALID, SAME}'.format(padding))
    stride = [1, stride, stride, 1]
    with tf.variable_scope(name):
        if act is not None:
            X = act(X)

        params = get_weights(filt_shape)
        X = tf.nn.conv2d_transpose(X, params, out_shape, stride, padding)

        if norm is None:
            bias_dim = [filt_shape[-2]]
            X = tf.nn.bias_add(X, get_bias(bias_dim))
        elif norm == 'BN':
            is_training = (True if mode == "TRAIN" else False)
            X = slim.batch_norm(
                X, is_training=is_training, updates_collections=None)
        elif norm == 'IN':
            X = instance_norm(X)
        else:
            logger.error('Invalid normalization! Choose from {None, BN, IN}')

    return X


def dropout(x, keep_prob=0.5, scope='dropout'):
    with tf.variable_scope(scope):
        return tf.nn.dropout(x, keep_prob)


def fully_connected(name, X, dim, activation=tf.nn.relu):
    in_dim = np.prod(tf_static_shape(X)[1:])
    with tf.variable_scope(name):
        if activation is not None:
            X = activation(X)
        X = tf.reshape(X, [-1, in_dim])
        wshape = (in_dim, dim)
        params = get_weights(wshape)
        X = tf.matmul(X, params)
        X = tf.nn.bias_add(X, get_bias(dim))
    return X


def loss_l1(pred, gt):
    return tf.losses.absolute_difference(gt, pred, scope='loss_l1')


def loss_sigmoid_cross_entropy(pred, gt_segment):
    with tf.variable_scope('loss_sigmoid_cross_entropy'):
        pred = tf.expand_dims(tf.reshape(pred, [-1]), axis=1)
        gt_segment = tf.expand_dims(tf.reshape(gt_segment, [-1]), axis=1)
        return tf.losses.sigmoid_cross_entropy(gt_segment, pred)


def log_quaternion_loss_batch(predictions, labels, use_logging):
  """A helper function to compute the error between quaternions.
  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.
  Returns:
    A Tensor of size [batch_size], denoting the error between the quaternions.
  """
  assertions = []
  if use_logging:
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(predictions), [1]) - 1),
                    1e-4)),
            ['The l2 norm of each prediction quaternion vector should be 1.']))
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(labels), [1]) - 1), 1e-4)),
            ['The l2 norm of each label quaternion vector should be 1.']))

  with tf.control_dependencies(assertions):
    product = tf.multiply(predictions, labels)
  internal_dot_products = tf.reduce_sum(product, [1])

  if use_logging:
    internal_dot_products = tf.Print(
        internal_dot_products,
        [internal_dot_products, tf.shape(internal_dot_products)],
        'internal_dot_products:')

  logcost = tf.log(1e-4 + 1 - tf.abs(internal_dot_products))
  return logcost


def loss_log_quaternion(predictions, labels, batch_size, use_logging=True):
  """A helper function to compute the mean error between batches of quaternions.
  The caller is expected to add the loss to the graph.
  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.
  Returns:
    A Tensor of size 1, denoting the mean error between batches of quaternions.
  """
  logcost = log_quaternion_loss_batch(predictions, labels, use_logging)
  logcost = tf.reduce_sum(logcost, [0])
  logcost = tf.multiply(logcost, 1.0 / batch_size, name='log_quaternion_loss')
  if use_logging:
    logcost = tf.Print(
        logcost, [logcost], '[logcost]', name='log_quaternion_loss_print')
  return logcost


def loss_quat_dist(pred_quat, gt_quat, batch_size):
    dist = tf.diag_part(tf.matmul(pred_quat, tf.transpose(gt_quat)))
    dist = tf.acos(tf.abs(dist))
    dist = tf.scalar_mul(2, dist)
    # dist = tf.diag_part(tf.matmul(pred_quat, tf.transpose(gt_quat)))
    # dist = tf.scalar_mul(2, tf.square(dist))
    # dist = tf.acos(dist - 1.0)
    loss = tf.reduce_mean(dist)
    return loss

quat_losses = {'log_dist': loss_log_quaternion, 'dist': loss_quat_dist}


def form_image_grid(input_tensor, grid_shape, image_shape, num_channels):
    """Arrange a minibatch of images into a grid to form a single image.
    Args:
      input_tensor: Tensor. Minibatch of images to format, either 4D
          ([batch size, height, width, num_channels]) or flattened
          ([batch size, height * width * num_channels]).
      grid_shape: Sequence of int. The shape of the image grid,
          formatted as [grid_height, grid_width].
      image_shape: Sequence of int. The shape of a single image,
          formatted as [image_height, image_width].
      num_channels: int. The number of channels in an image.
    Returns:
      Tensor representing a single image in which the input images have been
      arranged into a grid.
    Raises:
      ValueError: The grid shape and minibatch size don't match, or the image
          shape and number of channels are incompatible with the input tensor.
    """
    if grid_shape[0] * grid_shape[1] != int(input_tensor.get_shape()[0]):
        raise ValueError('Grid shape incompatible with minibatch size.')
    if len(input_tensor.get_shape()) == 2:
        num_features = image_shape[0] * image_shape[1] * num_channels
        if int(input_tensor.get_shape()[1]) != num_features:
            raise ValueError(
                'Image shape and number of channels incompatible with '
                'input tensor.')
    elif len(input_tensor.get_shape()) == 4:
        if (int(input_tensor.get_shape()[1]) != image_shape[0] or
                int(input_tensor.get_shape()[2]) != image_shape[1] or
                int(input_tensor.get_shape()[3]) != num_channels):
            raise ValueError(
                'Image shape and number of channels incompatible with'
                'input tensor.')
    else:
        raise ValueError('Unrecognized input tensor format.')
    height, width = grid_shape[0] * \
        image_shape[0], grid_shape[1] * image_shape[1]
    input_tensor = tf.reshape(input_tensor,
                              grid_shape + image_shape + [num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 1, 3, 2, 4])
    input_tensor = tf.reshape(
        input_tensor, [grid_shape[0], width, image_shape[0], num_channels])
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, 3])
    input_tensor = tf.reshape(input_tensor, [1, height, width, num_channels])
    return input_tensor


def im_views(ims, gh, gw, scope='im_views'):
    with tf.variable_scope(scope):
        _, _, h, w, ch = tf_static_shape(ims)
        im_grid = form_image_grid(collapse_dims(ims), [gh, gw], [h, w], ch)
        return tf.cast(im_grid * 255, tf.uint8)


def seg_views(ims, gh, gw, scope='seg_views'):
    with tf.variable_scope(scope):
        _, h, w, ch = tf_static_shape(ims)
        im_grid = form_image_grid(ims, [gh, gw], [h, w], ch)
        return tf.cast(im_grid * 255, tf.uint8)


def image_sum(im_tensor, nh, nw, tag='views'):
    return tf.summary.image(tag + '_sum', im_views(im_tensor, nh, nw, tag))


def segment_sum(net, pred_map, gt_map, tag=''):
    seg_sum = []
    if len(gt_map.get_shape().as_list()) is 2:
        gt_map = tf.expand_dims(gt_map, 0)
    pred_views = seg_views(vis_segment(pred_map), net.batch_size, 1, scope=tag+'seg_pred')
    gt_views = seg_views(vis_segment(gt_map), net.batch_size, 1, scope=tag+'seg_gt')
    # return tf.summary.image(tag + '_gt', gt_views)
    with tf.name_scope(tag+'seg_views'):
        view = tf.concat([gt_views, pred_views], axis=2)
        seg_sum.append(tf.summary.image(tag+'seg_view', view))
        return tf.summary.merge(seg_sum)


def segment_sum_test(net, pred_map, gt_map, thresh=[0.1], tag=''):
    with tf.name_scope('segment_views'+tag):
        seg_sum = []
        # pred_map = net.prob_seg_map
        # gt_map = net.gt_segment
        if len(gt_map.get_shape().as_list()) is 2:
            gt_map = tf.expand_dims(gt_map, 0)
        gt_views = seg_views(vis_segment(gt_map), net.batch_size, 1, scope='seg_gt'+tag)
        views = gt_views
        for th in thresh:
            seg_map = tf.greater(pred_map, th)
            seg_map = tf.to_float(seg_map)
            pred_views = seg_views(vis_segment(seg_map), net.batch_size, 1, scope='seg_pred'+tag+str(th))
            # return tf.summary.image(tag + '_gt', gt_views)
            views = tf.concat([views, pred_views], axis=2)
        thresh_tf = tf.convert_to_tensor([str(v) for v in thresh], dtype=tf.string)
        seg_sum.append(tf.summary.text('seg_thresh'+tag, thresh_tf))
        seg_sum.append(tf.summary.image('seg_view'+tag, views))
        return tf.summary.merge(seg_sum)


def vis_segment(s):
    with tf.name_scope('vis_segment'):
        s_v = tf.expand_dims(s, -1)
        # s_alpha = tf.to_float(s_v)
        # s_v = tf.concat([s_v, s_v, s_v, s_alpha], axis=-1)
        s_v = tf.concat([s_v, s_v, s_v], axis=-1)
        return s_v


def collapse_dims(T):
    shape = tf_static_shape(T)
    return tf.reshape(T, [-1] + shape[2:])


def uncollapse_dims(T, s1, s2):
    shape = tf_static_shape(T)
    return tf.reshape(T, [s1, s2] + shape[1:])


def tf_static_shape(T):
    return T.get_shape().as_list()