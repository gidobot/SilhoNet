import argparse
import logging
import os
import os.path as osp
import time
from pprint import pprint
import numpy as np
from math import acos

import tensorflow as tf
from tqdm import tqdm

from config import YCB_PATH
from models import grid_nets, im_nets, seg_nets, pose_nets, model_tm2sm_grid
from silhonet import SilhoNet
from ops import conv_rnns, image_sum, segment_sum, loss_ce, loss_sgm_ce, repeat_tensor, voxel_sum, loss_vp_triplet, loss_vp_triplet_cos_distance
from ycb import YCB
from utils import (Timer, get_session_config, init_logging, mkdir_p,
                   process_args, write_args)


# import lib.average_distance_loss.average_distance_loss_op as average_distance_loss_op
# import lib.average_distance_loss.average_distance_loss_op_grad

# Set visible GPU for testing
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def test_single_conv1d(sess):
	bs = 2
	gs = 3
	fs = 2
	iw = 3
	ih = 3

	W = np.random.rand(bs,gs,gs,gs,fs)
	X = np.random.rand(bs,iw,ih,fs)

	# tf way
	Wt = tf.Variable(W, dtype=tf.float32)
	Xt = tf.Variable(X, dtype=tf.float32)

	Wb = tf.reshape(Wt, [bs,gs**3,fs])
	Wb = tf.matrix_transpose(Wb)
	Wb = tf.expand_dims(Wb,1)

	Xb = tf.reshape(Xt, [bs, iw*ih, fs])

	def single_conv1d(tupl):
	    x, kernel = tupl
	    return tf.nn.conv1d(x, kernel, stride=1, padding='SAME')

	out_tf = tf.squeeze(tf.map_fn(
	    single_conv1d, (tf.expand_dims(Xb,1), Wb), dtype=tf.float32, name='conv_batch'),
	    axis=1)
	out_tf = tf.reshape(out_tf, [bs,iw,ih,gs**3])

	# np way
	Xn = X
	Wn = np.reshape(W,(bs,gs**3,fs))
	out_np = np.zeros((bs,iw,ih,gs**3))
	for b in range(bs):
		for w in range(iw):
			for h in range(ih):
				for g in range(gs**3):
					out_np[b,w,h,g] = np.dot(Wn[b,g,:], Xn[b,w,h,:])


	diff = tf.reduce_sum(tf.subtract(np.float32(out_np), out_tf))
	diff = tf.Print(diff, [diff])

	# run tests
	init = tf.global_variables_initializer()
	sess.run(init)
	sess.run(diff)

def test_average_distance_loss_op(sess):
	idx1 = 2
	idx2 = 4

	pose_target = tf.Variable(np.array([[1,0,0,0],[1,0,0,0]]), dtype=tf.float32)
	pose_pred = tf.Variable(np.array([[0.918,0.398,0,0],[1,0,0,0]]), dtype=tf.float32)

	points_all = np.load("points_mat.npy")

	points1 = points_all[idx1,:,:]/np.max(points_all[idx1,:,:])
	points2 = points_all[idx2,:,:]/np.max(points_all[idx2,:,:])
	points = tf.Variable(np.stack((points1,points2)), dtype=tf.float32)

	pose_weights = tf.Variable(np.array([[1,1,1,1],[1,1,1,1]]), dtype=tf.float32)
	symmetry = tf.Variable(np.array([0,0]), dtype=tf.float32)

	loss = average_distance_loss_op.average_distance_loss(pose_pred, pose_target, pose_weights, points, symmetry)

	out = tf.Print(loss[0], [loss[0]])

	# run tests
	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	sess.run(out)

def test_loss_vp_triplet(sess):
	# a = np.random.rand(1,64)
	# b = np.random.rand(1,1,64)
	# c = np.random.rand(1,64)
	a = np.array([[2,2,2,2]])
	b = np.array([[[1,2,3,4]]])
	c = np.array([[1,-1,1,-1]])
	a_tf = tf.Variable(a, dtype=tf.float64)
	b_tf = tf.Variable(b, dtype=tf.float64)
	c_tf = tf.Variable(c, dtype=tf.float64)
	margin = 1.0

	init = tf.global_variables_initializer()
	sess.run(init)

	tf_loss = loss_vp_triplet(a_tf,b_tf,c_tf,margin)

	loss_vec = np.zeros(1)
	# use numpy to calculate loss
	for i in range(a.shape[0]):
		bb_dot = np.dot(b[i], np.transpose(b[i]))
		bb_dot = np.diagonal(bb_dot)
		ba_dot = np.dot(b[i], np.transpose(a[i]))
		aa_dot = np.dot(a[i], a[i])
		cc_dot = np.dot(c[i], c[i])
		ac_dot = np.dot(a[i], c[i])
		neg_dist = np.min(bb_dot + aa_dot - 2.0*ba_dot)
		neg_dist = neg_dist.clip(min=0)
		pos_dist = cc_dot + aa_dot - 2.0*ac_dot
		pos_dist = pos_dist.clip(min=0)
		loss_vec[i] = pos_dist - neg_dist + margin
		loss_vec[i] = loss_vec[i].clip(min=0)
	np_loss = np.mean(loss_vec)

	print("TF loss: {}".format(tf_loss.eval()))
	print("np loss: {}".format(np_loss))
	import pdb
	pdb.set_trace()

def test_loss_vp_triplet_cos_dist(sess):
	# a = np.random.rand(6,64)    # predicted
	# b = np.random.rand(6,30,64) # samples
	# c = np.random.rand(6,64)	# ground truth
	a = np.array([[2,2,2,2]])
	b = np.array([[[1,2,3,4]]])
	c = np.array([[1,-1,1,-1]])
	a_quat = np.random.rand(1,4)
	for i in range(a_quat.shape[0]):
		a_quat[i,:] = a_quat[i,:] / np.linalg.norm(a_quat[i,:])
	b_quat = np.random.rand(1,1,4)
	for i in range(b_quat.shape[0]):
		for j in range(b_quat.shape[1]):
			b_quat[i,j,:] = b_quat[i,j,:] / np.linalg.norm(b_quat[i,j,:])
	c_quat = np.random.rand(1,4)
	for i in range(c_quat.shape[0]):
		c_quat[i,:] = c_quat[i,:] / np.linalg.norm(c_quat[i,:])
	a_tf = tf.Variable(a, dtype=tf.float64)
	b_tf = tf.Variable(b, dtype=tf.float64)
	c_tf = tf.Variable(c, dtype=tf.float64)
	a_quat_tf = tf.Variable(a_quat, dtype=tf.float64)
	b_quat_tf = tf.Variable(b_quat, dtype=tf.float64)
	c_quat_tf = tf.Variable(c_quat, dtype=tf.float64)

	init = tf.global_variables_initializer()
	sess.run(init)

	tf_loss = loss_vp_triplet_cos_distance(a_tf,b_tf,c_tf,0,1.0,a_quat_tf,b_quat_tf,c_quat_tf)

	loss_vec = np.zeros(1)
	# use numpy to calculate loss
	for i in range(a.shape[0]):
		bb_dot = np.dot(b[i], np.transpose(b[i]))
		bb_dot = np.diagonal(bb_dot)
		ba_dot = np.dot(b[i], np.transpose(a[i]))
		aa_dot = np.dot(a[i], a[i])
		cc_dot = np.dot(c[i], c[i])
		ac_dot = np.dot(a[i], c[i])
		neg_dist = np.min(bb_dot + aa_dot - 2.0*ba_dot)
		neg_dist = neg_dist.clip(min=0)
		pos_dist = cc_dot + aa_dot - 2.0*ac_dot
		pos_dist = pos_dist.clip(min=0)
		neg_idx = np.argmin(bb_dot + aa_dot - 2.0*ba_dot)
		aq = a_quat[i,:]
		bq = b_quat[i,neg_idx,:]
		cq = c_quat[i,:]
		pos_margin = 2*acos(np.dot(aq,cq))
		neg_margin = 2*acos(np.dot(aq,bq))
		margin = neg_margin - pos_margin
		# loss_vec[i] = 1 - (neg_dist/(pos_dist+margin))
		loss_vec[i] = pos_dist - neg_dist + margin
		loss_vec[i] = loss_vec[i].clip(min=0)
	np_loss = np.mean(loss_vec)

	print("TF loss: {}".format(tf_loss.eval()))
	print("np loss: {}".format(np_loss))

	import pdb
	pdb.set_trace()

if __name__ == '__main__':
	sess = tf.InteractiveSession()
	# test_single_conv1d()
	# test_average_distance_loss_op()
	# test_loss_vp_triplet(sess)
	test_loss_vp_triplet_cos_dist(sess)