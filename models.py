import logging
import tensorflow as tf
import numpy as np
import os.path as osp

from ops import (collapse_dims, conv2d, conv3d, deconv2d,
                 tf_static_shape, uncollapse_dims, dropout,
                 fully_connected, deconv_pcnn)


#####################################
##### Pretrained model loaders #####
#####################################

def load_pretrained_im_vgg16(net, sess):
    with tf.variable_scope('MVNet/ImNet_UNet', reuse=True):
        weights_path = 'data/imagenet_weights/vgg16_weights.npz'
        assert osp.exists(weights_path), \
                'VGG16 imagenet weights path does not exist: {}'.format(weights_path)
        weights = np.load('data/imagenet_weights/vgg16_weights.npz')
        sess.run(tf.get_variable('conv1_1/weights').assign(weights['conv1_1_W']))
        sess.run(tf.get_variable('conv1_1/bias').assign(weights['conv1_1_b']))
        sess.run(tf.get_variable('conv1_2/weights').assign(weights['conv1_2_W']))
        sess.run(tf.get_variable('conv1_2/bias').assign(weights['conv1_2_b']))
        sess.run(tf.get_variable('conv2_1/weights').assign(weights['conv2_1_W']))
        sess.run(tf.get_variable('conv2_1/bias').assign(weights['conv2_1_b']))
        sess.run(tf.get_variable('conv2_2/weights').assign(weights['conv2_2_W']))
        sess.run(tf.get_variable('conv2_2/bias').assign(weights['conv2_2_b']))
        sess.run(tf.get_variable('conv3_1/weights').assign(weights['conv3_1_W']))
        sess.run(tf.get_variable('conv3_1/bias').assign(weights['conv3_1_b']))
        sess.run(tf.get_variable('conv3_2/weights').assign(weights['conv3_2_W']))
        sess.run(tf.get_variable('conv3_2/bias').assign(weights['conv3_2_b']))
        sess.run(tf.get_variable('conv3_3/weights').assign(weights['conv3_3_W']))
        sess.run(tf.get_variable('conv3_3/bias').assign(weights['conv3_3_b']))
        sess.run(tf.get_variable('conv4_1/weights').assign(weights['conv4_1_W']))
        sess.run(tf.get_variable('conv4_1/bias').assign(weights['conv4_1_b']))
        sess.run(tf.get_variable('conv4_2/weights').assign(weights['conv4_2_W']))
        sess.run(tf.get_variable('conv4_2/bias').assign(weights['conv4_2_b']))
        sess.run(tf.get_variable('conv4_3/weights').assign(weights['conv4_3_W']))
        sess.run(tf.get_variable('conv4_3/bias').assign(weights['conv4_3_b']))
        sess.run(tf.get_variable('conv5_1/weights').assign(weights['conv5_1_W']))
        sess.run(tf.get_variable('conv5_1/bias').assign(weights['conv5_1_b']))
        sess.run(tf.get_variable('conv5_2/weights').assign(weights['conv5_2_W']))
        sess.run(tf.get_variable('conv5_2/bias').assign(weights['conv5_2_b']))
        sess.run(tf.get_variable('conv5_3/weights').assign(weights['conv5_3_W']))
        sess.run(tf.get_variable('conv5_3/bias').assign(weights['conv5_3_b']))
    return net

#####################################
##### Image processing networks #####
#####################################

def im_vgg16(net, ims):
    net.im_net = {}
    bs, h, w, ch = tf_static_shape(ims)
    with tf.variable_scope('ImNet_UNet', reuse=tf.AUTO_REUSE):
        #VGG16 layers
        conv1_1 = conv2d('conv1_1', ims, 3, 64, mode=net.mode, act=None)
        net.im_net['conv1_1'] = conv1_1
        conv1_2 = conv2d('conv1_2', conv1_1, 3, 64, mode=net.mode)
        net.im_net['conv1_2'] = conv1_2
        pool1 = tf.layers.max_pooling2d(conv1_2, 2, 2, padding='same', name='pool1')
        conv2_1 = conv2d('conv2_1', pool1, 3, 128, mode=net.mode)
        net.im_net['conv2_1'] = conv2_1
        conv2_2 = conv2d('conv2_2', conv2_1, 3, 128, mode=net.mode)
        net.im_net['conv2_2'] = conv2_2
        pool2 = tf.layers.max_pooling2d(conv2_2, 2, 2, padding='same', name='pool2')
        net.im_net['pool2'] = pool2
        conv3_1 = conv2d('conv3_1', pool2, 3, 256, mode=net.mode)
        net.im_net['conv3_1'] = conv3_1
        conv3_2 = conv2d('conv3_2', conv3_1, 3, 256, mode=net.mode)
        net.im_net['conv3_2'] = conv3_2
        conv3_3 = conv2d('conv3_3', conv3_2, 3, 256, mode=net.mode)
        net.im_net['conv3_3'] = conv3_3
        pool3 = tf.layers.max_pooling2d(conv3_3, 2, 2, padding='same', name='pool3')
        net.im_net['pool3'] = pool3
        conv4_1 = conv2d('conv4_1', pool3, 3, 512, mode=net.mode)
        net.im_net['conv4_1'] = conv4_1
        conv4_2 = conv2d('conv4_2', conv4_1, 3, 512, mode=net.mode)
        net.im_net['conv4_2'] = conv4_2
        conv4_3 = conv2d('conv4_3', conv4_2, 3, 512, mode=net.mode)
        net.im_net['conv4_3'] = conv4_3
        pool4 = tf.layers.max_pooling2d(conv4_3, 2, 2, padding='same', name='pool4')
        net.im_net['pool4'] = pool4
        conv5_1 = conv2d('conv5_1', pool4, 3, 512, mode=net.mode)
        net.im_net['conv5_1'] = conv5_1
        conv5_2 = conv2d('conv5_2', conv5_1, 3, 512, mode=net.mode)
        net.im_net['conv5_2'] = conv5_2
        conv5_3 = conv2d('conv5_3', conv5_2, 3, 512, mode=net.mode)
        net.im_net['conv5_3'] = conv5_3
        #Deconv layers
        feat_conv5   = conv2d('feat_conv5', conv5_3, 1, 64, norm=net.norm, mode=net.mode)
        net.im_net['feat_conv5'] = feat_conv5
        upfeat_conv5 = deconv_pcnn(feat_conv5, 4, 4, 64, 2, 2, name='upfeat_conv5', trainable=False)
        # upfeat_conv5 = deconv2d('upfeat_conv5', conv5_3, 4, 64, stride=2, padding="SAME", norm=net.norm, mode=net.mode)
        net.im_net['upfeat_conv5'] = upfeat_conv5
        feat_conv4   = conv2d('feat_conv4', conv4_3, 1, 64, norm=net.norm, mode=net.mode)
        net.im_net['feat_conv4'] = feat_conv4
        add_feat = tf.add_n([upfeat_conv5, feat_conv4], name='add_feat')
        add_feat = dropout(add_feat, net.keep_prob)
        net.im_net['add_feat'] = add_feat
        upfeat   = deconv_pcnn(add_feat, 16, 16, 64, 8, 8, name='upfeat', trainable=False)
        # upfeat = deconv2d('upfeat', add_feat, 16, 64, stride=8, padding="SAME", norm=net.norm, mode=net.mode)
        net.im_net['upfeat'] = upfeat
    return upfeat

im_nets = {'vgg16': im_vgg16}

###############################################
##### Model feature map resizing networks #####
###############################################

def mdl_resize_net_64(net, mdl_feats):
    # Assume input feature map dimensions of 224x224
    net.mdl_resize_net = {}
    with tf.variable_scope('MDLResize_Net'):
        # Output (bs, 112, 112)
        # conv1 = conv2d('conv1', mdl_feats, 3, 128, stride=2, padding="SAME", norm=net.norm, mode=net.mode)
        # net.mdl_resize_net['conv1'] = conv1
        # Output (bs, 56, 56)
        pool1 = tf.layers.max_pooling2d(mdl_feats, 4, 4, padding='same', name='pool1')
        net.mdl_resize_net['pool1'] = pool1
        # Output (bs, 60, 60)
        deconv1 = deconv2d('deconv1', pool1, 5, 64, stride=1, padding="VALID", norm=net.norm, mode=net.mode)
        net.mdl_resize_net['deconv1'] = deconv1
        # Output (bs, 64, 64)
        deconv2 = deconv2d('deconv2', deconv1, 5, 32, stride=1, padding="VALID", norm=net.norm, mode=net.mode)
        net.mdl_resize_net['deconv2'] = deconv2
    return deconv2

##########################################
##### Silhouette prediction networks #####
##########################################

def seg_stack(net, mdl_im_feats, roi_im_feats, scope='ImSeg_Net'):
    mdl_im_bs = net.mdl_im_tensor_shape[1]
    with tf.variable_scope(scope):
        mdl_im_feats = uncollapse_dims(mdl_im_feats, net.batch_size, mdl_im_bs)
        cat_mdl_feats = mdl_im_feats[:,0,:,:,:]
        for v in range(1,mdl_im_bs):
            cat_mdl_feats = tf.concat([cat_mdl_feats, mdl_im_feats[:,v,:,:,:]], -1)
        stack = tf.concat([cat_mdl_feats, roi_im_feats], -1)
        net.seg_net[scope+'_stack'] = stack
        conv1 = conv2d('conv1', stack, 2, 1024, stride=1, norm=net.norm, mode=net.mode)
        net.seg_net[scope+'_conv1'] = conv1
        conv2 = conv2d('conv2', conv1, 2, 512, stride=2, norm=net.norm, mode=net.mode)
        net.seg_net[scope+'_conv2'] = conv2
        conv3 = conv2d('conv3', conv2, 3, 256, stride=1, norm=net.norm, mode=net.mode)
        net.seg_net[scope+'_conv3'] = conv3
        conv4 = conv2d('conv4', conv3, 3, 256, stride=1, norm=net.norm, mode=net.mode)
        conv4 = dropout(conv4, net.keep_prob)
        net.seg_net[scope+'_conv4'] = conv4
        deconv1 = deconv2d('deconv1', conv4, 2, 256, stride=2, norm=net.norm, mode=net.mode)
        net.seg_net[scope+'_deconv1'] = deconv1
        out = conv2d('out', deconv1, 1, 1, stride=1, norm=net.norm, mode=net.mode)
        net.seg_net[scope+'_out'] = out
    return out

seg_nets = {'stack': seg_stack}

##################################################
##### 3D pose quaternion prediction networks #####
##################################################

# ResNet18 architecture
def quat_res(net, vp_mask):
    net.quat_net = {}
    with tf.variable_scope('Quat_Net', reuse=tf.AUTO_REUSE):
        vp_mask = tf.expand_dims(vp_mask, -1)
        # Output (bs, 32, 32, 64)
        conv1 = conv2d('conv1', vp_mask, 7, 64, stride=2, norm=net.norm, mode=net.mode, act=None)
        net.quat_net['conv1'] = conv1
        # Output (bs, 16, 16, 64)
        pool1 = tf.layers.max_pooling2d(conv1, 3, 2, padding='same', name='pool1')
        net.quat_net['pool1'] = pool1

        # Output (bs, 16, 16, 64)
        conv2_1a = conv2d('conv2_1a', pool1, 3, 64, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv2_1a'] = conv2_1a
        conv2_2a = conv2d('conv2_2a', conv2_1a, 3, 64, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv2_2a'] = conv2_2a
        res_2a = tf.add_n([conv2_2a, pool1], name='res_2a')

        conv2_1b = conv2d('conv2_1b', res_2a, 3, 64, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv2_1b'] = conv2_1b
        conv2_2b = conv2d('conv2_2b', conv2_1b, 3, 64, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv2_2b'] = conv2_2b
        res_2b = tf.add_n([conv2_2b, res_2a], name='res_2b')

        # Output (bs, 8, 8, 128)
        conv3_1a = conv2d('conv3_1a', res_2b, 3, 128, stride=2, norm=net.norm, mode=net.mode)
        net.quat_net['conv3_1a'] = conv3_1a
        conv3_2a = conv2d('conv3_2a', conv3_1a, 3, 128, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv3_2a'] = conv3_2a
        res_2b_skip = conv2d('res_2b_skip', res_2b, 1, 128, stride=2, norm=net.norm, mode=net.mode)
        res_3a = tf.add_n([conv3_2a, res_2b_skip], name='res_3a')

        conv3_1b = conv2d('conv3_1b', res_3a, 3, 128, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv3_1b'] = conv3_1b
        conv3_2b = conv2d('conv3_2b', conv3_1b, 3, 128, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv3_2b'] = conv3_2b
        res_3b = tf.add_n([conv3_2b, res_3a], name='res_3b')

        # Output (bs, 4, 4, 256)
        conv4_1a = conv2d('conv4_1a', res_3b, 3, 256, stride=2, norm=net.norm, mode=net.mode)
        net.quat_net['conv4_1a'] = conv4_1a
        conv4_2a = conv2d('conv4_2a', conv4_1a, 3, 256, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv4_2a'] = conv4_2a
        res_3b_skip = conv2d('res_3b_skip', res_3b, 1, 256, stride=2, norm=net.norm, mode=net.mode)
        res_4a = tf.add_n([conv4_2a, res_3b_skip], name='res_4a')

        conv4_1b = conv2d('conv4_1b', res_4a, 3, 256, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv4_1b'] = conv4_1b
        conv4_2b = conv2d('conv4_2b', conv4_1b, 3, 256, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv4_2b'] = conv4_2b
        res_4b = tf.add_n([conv4_2b, res_4a], name='res_4b')

        # Output (bs, 2, 2, 512)
        conv5_1a = conv2d('con5_1a', res_4b, 3, 512, stride=2, norm=net.norm, mode=net.mode)
        net.quat_net['con5_1a'] = conv5_1a
        conv5_2a = conv2d('con5_2a', conv5_1a, 3, 512, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['con5_2a'] = conv5_2a
        res_4b_skip = conv2d('res_4b_skip', res_4b, 1, 512, stride=2, norm=net.norm, mode=net.mode)
        res_5a = tf.add_n([conv5_2a, res_4b_skip], name='res_5a')

        conv5_1b = conv2d('conv5_1b', res_5a, 3, 512, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv5_1b'] = conv5_1b
        conv5_2b = conv2d('conv5_2b', conv5_1b, 3, 512, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv5_2b'] = conv5_2b
        res_5b = tf.add_n([conv5_2b, res_5a], name='res_5b')
        res_5b = dropout(res_5b, net.keep_prob)

        # Output (bs, 4*num_classes)
        fc1 = fully_connected('fc1', res_5b, 512)
        net.quat_net['fc1'] = fc1
        fc2 = fully_connected('fc2', fc1, 4*net.num_classes)
        net.quat_net['fc2'] = fc2
        # out = tf.tanh(fc2)
        out = fc2
        net.quat_net['out'] = out

    return out

# InceptionNet inspired
def quat_inception(net, vp_mask):
    net.quat_net = {}
    with tf.variable_scope('Viewpoint_Net', reuse=tf.AUTO_REUSE):
        vp_mask = tf.expand_dims(vp_mask, -1)
        # Output (bs, 64, 64, ch)
        conv1 = conv2d('conv1', vp_mask, 3, 256, stride=1, norm=net.norm, mode=net.mode, act=None)
        net.quat_net['conv1'] = conv1
        conv2 = conv2d('conv2', conv1, 1, 128, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv2'] = conv2
        conv3 = conv2d('conv3', conv2, 1, 128, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv3'] = conv3
        # Output (bs, 32, 32, ch)
        pool1 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same', name='pool1')
        net.quat_net['pool1'] = pool1
        conv4 = conv2d('conv4', pool1, 3, 512, stride=1, norm=net.norm, mode=net.mode)
        net.quat_net['conv4'] = conv4
        conv5 = conv2d('conv5', conv4, 1, 256, stride=1, norm=net.norm, mode=net.mode)
        conv5 = dropout(conv5, net.keep_prob)
        net.quat_net['conv5'] = conv5
        # Output (bs, 16, 16, ch)
        pool2 = tf.layers.max_pooling2d(conv5, 2, 2, padding='same', name='pool2')
        pool2 = dropout(pool2, net.keep_prob)
        net.quat_net['pool2'] = pool2
        fc1 = fully_connected('fc1', pool2, 1024)
        net.quat_net['fc1'] = fc1
        fc2 = fully_connected('fc2', fc1, 4*net.num_classes)
        # fc2 = tf.tanh(fc2)
        net.quat_net['fc2'] = fc2
        out = fc2
        net.quat_net['out'] = out
    return out

quat_nets = {'quat_res': quat_res, 'quat_inception': quat_inception}

########################################
###### SilhoNet graph definitions ######
########################################

# Silhouette prediction graph
def model_silhouette(net,
               im_net=im_vgg16,
               seg_net=seg_stack,
               threshold_mask=False):
    ''' Silhouette prediction model '''
    with tf.variable_scope('MVNet'):
        # Compute model image features
        net.mdl_ims_stack = collapse_dims(net.mdl_ims)
        num_its = int(tf_static_shape(net.mdl_ims_stack)[0] / net.batch_size)
        for v in range(0,num_its):
            im_feats = im_net(net, net.mdl_ims_stack[v*net.batch_size:(1+v)*net.batch_size,...])
            if v == 0:
                net.mdl_im_feats = im_feats
            else:
                net.mdl_im_feats = tf.concat([net.mdl_im_feats, im_feats], 0)
        # Compute input image features
        net.input_im_feats = im_net(net, net.input_ims)
        # Extract ROIs from input image feature maps and resize to fixed size
        net.roi_im_feats = tf.image.crop_and_resize(net.input_im_feats,
                                                    net.rois,
                                                    tf.range(tf_static_shape(net.rois)[0]),
                                                    net.roi_shape[1:],
                                                    method='bilinear')
        # Use convolution layers to resize each mdl image feature map to 64x64
        net.mdl_resized_feats = mdl_resize_net_64(net, net.mdl_im_feats)
        # Compute segmentation maps
        net.seg_net = {}
        net.pred_seg_map = seg_net(net, net.mdl_resized_feats, net.roi_im_feats, scope='im_seg_net')
        net.pred_seg_map = tf.squeeze(net.pred_seg_map, [-1])
        net.prob_seg_map = tf.nn.sigmoid(net.pred_seg_map)
        if threshold_mask:
            net.prob_seg_map = tf.cast((net.prob_seg_map > 0.3), tf.float32)
        net.pred_occ_seg_map = seg_net(net, net.mdl_resized_feats, net.roi_im_feats, scope='im_occ_seg_net')
        net.pred_occ_seg_map = tf.squeeze(net.pred_occ_seg_map, [-1])
        net.prob_occ_seg_map = tf.nn.sigmoid(net.pred_occ_seg_map)
        if threshold_mask:
            net.prob_occ_seg_map = tf.cast((net.prob_occ_seg_map > 0.3), tf.float32)
        return net

# 3D pose quaternion prediction graph
def model_quat(net, quat_net=quat_res):
    ''' Quaternion prediction model '''
    with tf.variable_scope('Quaternion_TM2SM'):
        if net.training:
            silhouette = net.gt_segment
        else:
            silhouette = tf.stop_gradient(net.prob_seg_map)
        net.pred_quat = quat_net(net, silhouette)
        # extract predictions from class output vectors
        for v in range(0,net.batch_size):
            dim = int(tf_static_shape(net.pred_quat)[-1] / net.num_classes)
            cls_id = net.class_id[v]
            quat = net.pred_quat[v, cls_id*dim:(cls_id+1)*dim]
            quat = tf.expand_dims(quat, axis=0)
            if v == 0:
                quat_stack = quat
            else:
                quat_stack = tf.concat([quat_stack, quat], 0)
        net.pred_quat = quat_stack
        net.pred_quat = tf.nn.l2_normalize(net.pred_quat, dim=-1)
        return net