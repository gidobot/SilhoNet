import json
import logging
import os
import os.path as osp
import sys
import random
from queue import Empty, Queue
from threading import Thread, current_thread
from math import floor, ceil, acos
from random import shuffle
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import euler2mat, mat2euler, euler2quat

import scipy.misc
import scipy.io
import numpy as np
import glob

from loader import read_camera, read_depth, read_im, read_quat, read_classes, read_label, read_roi, read_poses, read_segments, \
                   read_viewpoint_ids, read_viewpoint_yaws, read_viewpoints, read_occluded_segments, read_viewpoint_quats, \
                   read_quat_viewpoint, load_pred_rois

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YCB_PATH = os.path.join(BASE_DIR, 'data/YCB')
COCO_PATH = os.path.join(BASE_DIR, 'data/COCO')

class YCB(object):
    def __init__(self,
                 image_set,
                 roi_size,
                 roi_area_thresh,
                 roi_fill_thresh,
                 shuffle=True,
                 rng_seed=0,
                 mode='train',
                 quat_regress=False,
                 use_pred_rois=False):
        self._mode = mode
        self._image_set = image_set
        self._ycb_path = YCB_PATH
        self._coco_path = COCO_PATH
        self._data_path = osp.join(self._ycb_path, 'data')
        self._model_path  = osp.join(self._ycb_path, 'models/rendered')
        assert osp.exists(self._ycb_path), \
                'ycb path does not exist: {}'.format(self._ycb_path)
        assert osp.exists(self._data_path), \
                'Data path does not exist: {}'.format(self._data_path)
        assert osp.exists(self._model_path), \
                'Model path does not exist: {}'.format(self._model_path)
        if mode is 'train':
            assert osp.exists(self._coco_path), \
                    'COCO path does not exist: {}'.format(self._coco_path)
            self._coco_file_list = os.listdir(self._coco_path)
        
        # For training quaternion net on rendered silhouette sets 
        # self._viewpoint_path = osp.join(self._ycb_path, 'models/rendered_viewpoints')
        # assert osp.exists(self._viewpoint_path), \
        #         'Viewpoint path does not exist: {}'.format(self._viewpoint_path)
        # self._quat_viewpoint_path = osp.join(self._ycb_path, 'models/rendered_quat_viewpoints')
        # assert osp.exists(self._quat_viewpoint_path), \
        #         'Quaternion viewpoint path does not exist: {}'.format(self._quat_viewpoint_path)
        # self._quat_viewpoint_list = self._load_viewpoint_quat_list(osp.join(self._quat_viewpoint_path, 'viewpoint_list.txt'))
        # if self.shuffle:
        #     self._shuffle_quat_viewpoints_list()
        self._quat_regress = quat_regress
        self.all_items_quat_viewpoints = ['segment','quat','class_id'] #,'viewpoint_quat','sample_vp_quats','sample_viewpoints']

        self.roi_size = roi_size
        self.roi_area_thresh = roi_area_thresh
        self.roi_fill_thresh = roi_fill_thresh

        self.shuffle = shuffle
        self.rng = rng_seed
        np.random.seed(self.rng)

        self.all_items = ['im','im_index','segment','occluded_segment','roi','im_mdl','quat','model_id','seq_id','class_id']

        self.logger = logging.getLogger('silhonet.' + __name__)

        self._classes = ('__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
                         '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '019_pitcher_base', \
                         '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', \
                         '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')

        self._num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self._num_classes)))

        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        if self.shuffle:
            self._shuffle_image_set_index()

        self._use_pred_rois = use_pred_rois
        if self._use_pred_rois:
            self._pred_roi_path = osp.join(BASE_DIR, 'data/frcnn_detections_ycb.json')
            assert osp.exists(self._pred_roi_path), \
                    'Predicted ROI path does not exist: {}'.format(self._pred_roi_path)
            self._pred_roi_dict = self._load_pred_rois(self._pred_roi_path)

        # Not used for SilhoNet
        # self._class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
        #                       (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
        #                       (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64), 
        #                       (192, 0, 0), (0, 192, 0), (0, 0, 192)]
        # self._class_weights = [1, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
        # self._symmetry = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        # self._extents = self._load_object_extents()
        # self._points, self._points_all = self._load_object_points()


    # image
    def get_im_ids(self):
        return self._image_index

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = osp.join(self._data_path, index + '-color' + self._image_ext)
        assert osp.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # label
    def label_path_at(self, i):
        """
        Return the absolute path to label i in the image sequence.
        """
        return self.label_path_from_index(self._image_index[i])

    def label_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        label_path = osp.join(self._data_path, index + '-label' + self._image_ext)
        assert osp.exists(label_path), \
                'Path does not exist: {}'.format(label_path)
        return label_path

    # segment
    def segment_path_at(self, i):
        """
        Return the absolute path to segmentation file i in the image sequence.
        """
        return self.label_path_from_index(self._image_index[i])

    def segment_path_from_index(self, index):
        """
        Construct a segmentation path from the image's "index" identifier.
        """
        segment_path = osp.join(self._data_path, index + '-segments.mat')
        assert osp.exists(segment_path), \
                'Path does not exist: {}'.format(segment_path)
        return segment_path

    def viewpoint_path_from_index(self, index):
        """
        Construct a viewpoint path from the classes's "index" identifier.
        """
        viewpoint_path = osp.join(self._viewpoint_path, self._classes[index], 'viewpoints.mat')
        assert osp.exists(viewpoint_path), \
                'Path does not exist: {}'.format(viewpoint_path)
        return viewpoint_path

    # camera pose
    def metadata_path_at(self, i):
        """
        Return the absolute path to metadata i in the image sequence.
        """
        return self.metadata_path_from_index(self._image_index[i])

    def metadata_path_from_index(self, index):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        metadata_path = osp.join(self._data_path, index + '-meta.mat')
        assert osp.exists(metadata_path), \
                'Path does not exist: {}'.format(metadata_path)
        return metadata_path

    # roi coordinates
    def roi_path_at(self, i):
        """
        Return the absolute path to roi i in the image sequence.
        """
        return self.roi_path_from_index(self._image_index[i])

    def roi_path_from_index(self, index):
        """
        Construct an roi path from the image's "index" identifier.
        """
        roi_path = osp.join(self._data_path, index + '-box.txt')
        assert osp.exists(roi_path), \
                'Path does not exist: {}'.format(roi_path)
        return roi_path

    def _load_viewpoint_quat_list(self, f):
        """
        Load the file paths listed in this dataset's viewpoint + quaternion set file.
        """
        vp_set_file = f
        assert osp.exists(vp_set_file), \
                'Path does not exist: {}'.format(f)

        with open(vp_set_file) as fd:
            index_list = [x.strip()[2:] for x in fd.readlines() if x.strip()]
        return index_list


    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = osp.join(self._ycb_path, 'image_sets', self._image_set + '.txt')
        assert osp.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines() if x.strip()]
        return image_index

    def _load_pred_rois(self, f):
        return load_pred_rois(f)


    def _load_object_extents(self):

        extent_file = osp.join(self._ycb_path, 'extents.txt')
        assert osp.exists(extent_file), \
                'Path does not exist: {}'.format(extent_file)

        extents = np.zeros((self._num_classes, 3), dtype=np.float32)
        extents[1:, :] = np.loadtxt(extent_file)

        return extents


    def _load_object_points(self):
        points = [[] for _ in range(self._num_classes)]
        num = np.inf

        for i in range(1, self._num_classes):
            point_file = osp.join(self._ycb_path, 'models', self._classes[i], 'points.xyz')
            print(point_file)
            assert osp.exists(point_file), 'Path does not exist: {}'.format(point_file)
            points[i] = np.loadtxt(point_file)
            if points[i].shape[0] < num:
                num = points[i].shape[0]
            # Rescale points
            weight = 2.0 / np.amax(self._extents[i, :])
            if weight < 10:
                weight = 10
            if self._symmetry[i] > 0:
                points[i] = 4 * weight * points[i]
            else:
                points[i] = weight * points[i]
            # points[i] = points[i]/np.max(points[i])

        points_all = np.zeros((self._num_classes, num, 3), dtype=np.float32)
        for i in range(1, self._num_classes):
            points_all[i, :, :] = points[i][:num, :]

        return points, points_all


    def _shuffle_image_set_index(self):
        shuffle(self._image_index)

    def _shuffle_quat_viewpoints_list(self):
        shuffle(self._quat_viewpoint_list)

    def _get_classes_from_index(self, index):
        # metadata path
        metadata_path = self.metadata_path_from_index(index)
        # get classes from metadata
        classes = read_classes(metadata_path)
        return classes

    def compute_class_weights(self):
        print('computing class weights')
        num_classes = self._num_classes
        count = np.zeros((num_classes,), dtype=np.int64)
        k = 0
        while k < len(self.image_index):
            index = self.image_index[k]
            # label path
            label_path = self.label_path_from_index(index)
            im = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            for i in xrange(num_classes):
                I = np.where(im == i)
                count[i] += len(I[0])
            k += 100

        count[0] = 0
        max_count = np.amax(count)

        for i in xrange(num_classes):
            if i == 0:
                self._class_weights[i] = 1
            else:
                self._class_weights[i] = min(2 * float(max_count) / float(count[i]), 10.0)
            print(self._classes[i], self._class_weights[i])

    def _load_ycb_annotation(self, index):
        """
        Load class name and meta data
        """
        # image path
        image_path = self.image_path_from_index(index)

        # label path
        label_path = self.label_path_from_index(index)

        # metadata path
        metadata_path = self.metadata_path_from_index(index)

        # segmentatoin path
        segment_path = self.segment_path_from_index(index)

        # roi path
        roi_path = self.roi_path_from_index(index)
        
        return {'image': image_path,
                'label': label_path,
                'meta_data': metadata_path,
                'segment': segment_path,
                'roi': roi_path,
                'class_colors': self._class_colors,
                'class_weights': self._class_weights,
                'cls_index': -1,
                'flipped': False}

    def _process_label_image(self, label_image):
        """
        change label image to label index
        """
        class_colors = self._class_colors
        width = label_image.shape[1]
        height = label_image.shape[0]
        label_index = np.zeros((height, width), dtype=np.float32)

        # label image is in BGR order
        index = label_image[:,:,2] + 256*label_image[:,:,1] + 256*256*label_image[:,:,0]
        for i in xrange(len(class_colors)):
            color = class_colors[i]
            ind = color[0] + 256*color[1] + 256*256*color[2]
            I = np.where(index == ind)
            label_index[I] = i

        return label_index

    def labels_to_image(self, im, labels):
        class_colors = self._class_colors
        height = labels.shape[0]
        width = labels.shape[1]
        image_r = np.zeros((height, width), dtype=np.float32)
        image_g = np.zeros((height, width), dtype=np.float32)
        image_b = np.zeros((height, width), dtype=np.float32)

        for i in xrange(len(class_colors)):
            color = class_colors[i]
            I = np.where(labels == i)
            image_r[I] = color[0]
            image_g[I] = color[1]
            image_b[I] = color[2]

        image = np.stack((image_r, image_g, image_b), axis=-1)

        return image.astype(np.uint8)

    def get_sid(self, im_index):
        return np.array([im_index.split('/')[0]])

    def get_im(self, im_index):
        f = self.image_path_from_index(im_index)
        train = self._mode == 'train'
        if 'data_syn' in f:
            im = read_im(f, train, osp.join(self._coco_path,random.choice(self._coco_file_list)))
        else:
            im = read_im(f, train)
        return im

    def get_label(self, im_index):
        f = self.label_path_from_index(im_index)
        im = read_label(f)
        return im

    def get_pose(self, im_index, cls_idx):
        f = self.metadata_path_from_index(im_index)
        poses = read_poses(f)
        pose = np.zeros((7), dtype=np.float32)
        for i, j in enumerate(poses[:,0]):
            if j == cls_idx:
                # pose = poses[i, 1:8] # [qw,qx,qy,qz,x,y,z]
                # TODO: only dealing with orientation for now
                pose = poses[i, 1:5] # [qw,qx,qy,qz]
                break
        return pose

    def get_segment(self, im_index, cls_idx):
        f = self.segment_path_from_index(im_index)
        segment_dict = read_segments(f)
        segment = segment_dict.get(cls_idx)
        return segment

    def get_viewpoint_id(self, im_index, cls_idx):
        f = self.segment_path_from_index(im_index)
        vp_id_dict = read_viewpoint_ids(f)
        vp_id = vp_id_dict.get(cls_idx)
        return vp_id

    def get_all_viewpoints(self, cls_idx):
        f = self.viewpoint_path_from_index(cls_idx)
        vp_dict = read_viewpoints(f)
        return vp_dict

    def get_viewpoint(self, cls_idx, vp_idx):
        f = self.viewpoint_path_from_index(cls_idx)
        vp_dict = read_viewpoints(f)
        vp = vp_dict.get(vp_idx)
        return vp

    def get_sample_viewpoints(self, cls_idx, vp_idx):
        vp_list = []
        quat_list = []
        if self._num_sample_viewpoints > 0:
            f = self.viewpoint_path_from_index(cls_idx)
            vp_dict = read_viewpoints(f)
            quat_dict = read_viewpoint_quats(f)
            num_vps = len(vp_dict)
            num_samples = min(self._num_sample_viewpoints, num_vps-1) # do not include vp_idx in choices
            indexes = random.sample([x for x in range(num_vps) if x != vp_idx], num_samples) # add filler samples if needed
            indexes = indexes + [indexes[0]]*(self._num_sample_viewpoints - num_samples)
            for idx in indexes:
                vp_list.append(vp_dict[idx])
                quat_list.append(quat_dict[idx])
        return np.array(vp_list), np.array(quat_list)

    def get_viewpoint_yaw(self, im_index, cls_idx):
        f = self.segment_path_from_index(im_index)
        vp_yaw_dict = read_viewpoint_yaws(f)
        vp_yaw = vp_yaw_dict.get(cls_idx)
        return vp_yaw

    def get_viewpoint_quat(self, cls_idx, vp_idx):
        f = self.viewpoint_path_from_index(cls_idx)
        vp_quat_dict = read_viewpoint_quats(f)
        vp_quat = vp_quat_dict.get(vp_idx)
        return vp_quat

    def get_all_viewpoint_quats(self, cls_idx):
        f = self.viewpoint_path_from_index(cls_idx)
        vp_quat_dict = read_viewpoint_quats(f)
        return vp_quat_dict

    # Hacky function for now to get viewpoint quaternion for triplet loss. Converted code from Matlab
    def get_relative_viewpoint_quat(self, q, class_idx, full=False):
        symmetry    =          [0, 4, 2, 2,  4,   1,  4,  2,  2, 1, 0, 0, 0,  3, 0, 0,    2, 0,  3,  1,   1, 2]
        rot_offsets = np.array([np.radians([0, 90, 0, 0, 90,   0, 90,  0,  0, 0, 0, 0, 0, 90, 0, 0,    0, 0,  0, 94,  90, 0]),
                                 np.radians([0, 0, 0, 0,  0,   0,  0,  0,  0, 0, 0, 0, 0,  0, 0, 0,    0, 0,  0,  9, -84, 0]),
                                np.radians([0, 90, 0, 0, 90, -22, 90, 28, 13, 4, 0, 0, 0, 90, 0, 0,  -12, 0, 92, -5,  -1, 0])])
        R = quat2mat(q) # no matrix inversion here
        rot_adj = euler2mat(rot_offsets[0,class_idx], rot_offsets[1,class_idx], rot_offsets[2,class_idx], axes='rxyz') # inverse taken in Matlab with SpinCalc
        pose = np.degrees(mat2euler(np.linalg.inv(np.dot(R,rot_adj)), axes='rxyz'))
        if pose[0] > 180:
            pose[0] = pose[0] - 360
        elif pose[0] < -180:
            pose[0] = pose[0] + 360
        if pose[1] > 180:
            pose[1] = pose[1] - 360
        elif pose[1] < -180:
            pose[1] = pose[1] + 360
        # 0 = no symmetry - angle ranges: roll = (-179.5,179.5), pitch = (-89.5,89.5)
        # 1 = planar symmetry - angle ranges: roll = (0.5,179.5), pitch = (-89.5,89.5)
        # 2 = 2 x planar symmetry - angle ranges: roll = (0.5,89.5), pitch = (-89.5,89.5)
        # 3 = infinite symmetry - angle ranges: roll = roll = 0, pitch = (-89.5,89.5)
        # 4 = infinite symmetry + planar symmetry: roll = 0, pitch = (0.5,89.5)
        roll = pose[0]
        pitch = pose[1]
        yaw = pose[2]
        if symmetry[class_idx] == 1:
            if roll < 0:
                roll = -roll
                pitch = -pitch
                yaw = yaw + 180
        elif symmetry[class_idx] == 2:
            if roll < 0:
                roll = -roll
                pitch = -pitch
                yaw = yaw + 180
            if roll > 90:
                roll = 180 - roll
                pitch = -pitch
                yaw = yaw + 180
        elif symmetry[class_idx] == 3:
            roll = 0
        elif symmetry[class_idx] == 4:
            roll = 0
            if pitch < 0:
                pitch = -pitch
                yaw = yaw + 180
        if yaw > 360:
            yaw = yaw - 360
        elif yaw < 0:
            yaw = yaw + 360

        if full:
            vp_quat = euler2quat(np.radians(roll), np.radians(pitch), np.radians(yaw), axes='rxyz')
        else:
            vp_quat = euler2quat(np.radians(roll), np.radians(pitch), 0, axes='rxyz')
        return vp_quat

    def get_occluded_segment(self, im_index, cls_idx):
        f = self.segment_path_from_index(im_index)
        segment_dict = read_occluded_segments(f)
        segment = segment_dict.get(cls_idx)
        return segment

    def get_symmetry(self, cls_idx):
        return self._symmetry[cls_idx]

    def get_points(self, cls_idx):
        return self._points_all[cls_idx, :, :]

    def get_roi(self, im_index):
        f = self.roi_path_from_index(im_index)
        roi_dict = read_roi(f)
        return roi_dict

    def get_model(self, im_index, cls_idx):
        ims = []
        f_list = glob.glob(osp.join(self._model_path, self._classes[cls_idx], '*_albedo.png'))
        for f in f_list:
            ims.append(read_im(f))
        return np.stack(ims, axis=0)

    def get_K(self, im_index, cls_idx):
        cams = []
        f_list = glob.glob(osp.join(self._model_path, self._classes[cls_idx], '*_camera.mat'))
        for f in f_list:
            cams.append(read_camera(f))
        camK = np.stack([c[0] for c in cams], axis=0)
        return camK

    def get_R(self, im_index, cls_idx):
        cams = []
        f_list = glob.glob(osp.join(self._model_path, self._classes[cls_idx], '*_camera.mat'))
        for f in f_list:
            cams.append(read_camera(f))
        camR = np.stack([c[1] for c in cams], axis=0)
        return camR

    def get_quat(self, im_index, cls_idx):
        cams = []
        f_list = glob.glob(osp.join(self._model_path, self._classes[cls_idx], '*_camera.mat'))
        for f in f_list:
            cams.append(read_quat(f))
        camq = np.stack(cams, axis=0)
        return camq

    # Expected roi coordinates are not normalized
    def get_roi_im(self, im, roi):
        crop = im[floor(roi[0]):ceil(roi[2]), floor(roi[1]):ceil(roi[3]), :]
        return crop

    # Expected roi coordinates are not normalized
    def get_roi_label(self, label, roi):
        crop = label[floor(roi[0]):ceil(roi[2]), floor(roi[1]):ceil(roi[3])]
        return crop


    def load_quat_viewpoint(self, index):
        file_name = self._quat_viewpoint_list[index]
        cls_name = file_name.split('/')[0]
        cls_idx = self._class_to_ind[cls_name]
        mask, quat = read_quat_viewpoint(osp.join(self._quat_viewpoint_path, file_name))
        data = {
            'segment': mask,
            'quat': quat,
            'class_id': cls_idx
        }
        data = [data]
        return data

    def load_quat_viewpoint_ycb(self, im_index):
        classes = self._get_classes_from_index(im_index)
        data_list = []
        for cls_idx in classes:
            data = {}
            cls_name = self._classes[cls_idx]
            q = self.get_pose(im_index, cls_idx) # only quaternion for now
            # get groundtruth quaternion with silhouette space symmetry reductions
            # and class coordinate frame corrections.
            quat = self.get_relative_viewpoint_quat(q, cls_idx, full=True)
            data = {
                'segment': self.get_segment(im_index, cls_idx),
                'quat': quat,
                'class_id': cls_idx
            }
            data_list.append(data.copy())
        return data_list


    def roi_intersection_area(self, roi_a, roi_b):  # returns None if rectangles don't intersect
        dx = min(roi_a[3], roi_b[3]) - max(roi_a[1], roi_b[1])
        dy = min(roi_a[2], roi_b[2]) - max(roi_a[0], roi_b[0])
        if (dx>=0) and (dy>=0):
            return dx*dy
        else:
            return None


    def load_ycb(self, im_index):
        classes = self._get_classes_from_index(im_index)
        im = self.get_im(im_index)
        label = self.get_label(im_index)
        roi_dict = self.get_roi(im_index)

        if self._use_pred_rois:
            if im_index in self._pred_roi_dict:
                pred_roi_list = self._pred_roi_dict[im_index]
            else:
                print("No predictions for this image index: {}".format(im_index))
                return []
        
        label_class_idx_list = np.unique(label)

        h, w = im.shape[:2]

        data_list = []
        for cls_idx in classes:
            data = {}
            cls_name = self._classes[cls_idx]
            if cls_name in roi_dict:
                roi = roi_dict.get(cls_name)
            else:
                continue

            if self._use_pred_rois:
                pred_roi_item = None
                pred_roi = None
                for index, d in enumerate(pred_roi_list):
                    if d['category_id'] == cls_idx:
                        if pred_roi_item is None:
                            pred_roi_item = d
                            pred_roi = d['roi']
                        else:
                            if d['score'] > pred_roi_item['score']:
                                pred_roi_item = d
                                pred_roi = d['roi']
                if pred_roi is None:
                    print("No matching predicted roi")
                    continue

            # Reject rois with reverse image bound issue
            if (roi[2] - roi[0]) < 0 or (roi[3] - roi[1]) < 0:
                # self.logger.debug('Skipping roi with reverse bounding')
                continue

            label_cls = label.copy()
            label_cls[label_cls != cls_idx] = 0
            label_cls[label_cls != 0] = 1

            roi_area = abs((roi[2] - roi[0]) * (roi[3] - roi[1]))
            if self._use_pred_rois:
                pred_intersection = self.roi_intersection_area(roi, pred_roi)
                if pred_intersection is None:
                    print("Predicted roi does not overlap by at least 50%.")
                    continue
                if pred_intersection > 0.5*roi_area:
                    roi = pred_roi
                    roi_area = abs((roi[2] - roi[0]) * (roi[3] - roi[1]))
                else:
                    print("Predicted roi does not overlap by at least 50%.")
                    continue

            # Note get_roi funcs take non-normalized coordinates
            label_roi = self.get_roi_label(label_cls, roi)

            # only consider rois with a significant area and limited occlusion
            # if roi_area < self.roi_area_thresh:
            if roi_area < self.roi_area_thresh or (float(np.sum(label_roi))/float(roi_area) < self.roi_fill_thresh):
                #self.logger.debug('Skipping roi for too small area or excessive occlusion')
                continue

            # normalize roi coordinates for tensorflow
            roi = roi / np.array([h, w, h, w])

            # get dataset label pose as quaternion
            q = self.get_pose(im_index, cls_idx) # only quaternion for now

            # get groundtruth quaternion with silhouette space symmetry reductions
            # and class coordinate frame corrections.
            quat = self.get_relative_viewpoint_quat(q, cls_idx, full=True)

            data = {
                'im': im.copy(),
                'im_index': im_index,
                'segment': self.get_segment(im_index, cls_idx),
                'occluded_segment': self.get_occluded_segment(im_index, cls_idx),
                'roi': roi, # with normalized coordinates
                'im_mdl': self.get_model(im_index, cls_idx),
                'quat': quat,
                'model_id': np.array([cls_name]),
                'seq_id': self.get_sid(im_index),
                'class_id': cls_idx # - 1 # don't include background class
            }
            data_list.append(data.copy())

        return data_list


    def test_load_ycb_all(self):
        for im_idx in range(len(self._image_index)):
            self.logger.debug('Testing image index {} of {}'.format(im_idx,len(self._image_index)))
            im_index = self._image_index[im_idx]
            self.load_ycb(im_index)

    def test_load_quat_viewpoint(self):
        # for idx in range(len(self._quat_viewpoint_list)):
        #     self.logger.debug('Testing viewpoint index {} of {}'.format(idx,len(self._quat_viewpoint_list)))
        #     self.load_quat_viewpoint(idx)
        for im_idx in range(len(self._image_index)):
            self.logger.debug('Testing image index {} of {}'.format(im_idx,len(self._image_index)))
            im_index = self._image_index[im_idx]
            self.load_quat_viewpoint_ycb(im_index)

    def fetch_data(self):
        # with self.coord.stop_on_exception():
        try:
            while not self.coord.should_stop():
                try:
                    idx = self.queue_idx.get(timeout=0.5)
                except Empty:
                    self.logger.debug('Index queue empty - {:s}'.format(
                        current_thread().name))
                    continue

                if self._quat_regress:
                    # data_list = self.load_quat_viewpoint(idx)
                    data_list = self.load_quat_viewpoint_ycb(self._image_index[idx])
                else:
                    data_list = self.load_ycb(self._image_index[idx])
                
                for data in data_list:
                    self.queue_data.put(data)
                    self.total_items += 1
                self.total_items -= 1

                if self.loop_data:
                    self.queue_idx.put(idx)
        except Exception as e:
            self.logger.debug(type(e))
            self.logger.debug(e.args)
            self.logger.debug(e)
            self.coord.request_stop(sys.exc_info())

    def init_queue(self,
                   coord,
                   nepochs=None,
                   qsize=32,
                   nthreads=4):
        self.coord = coord
        self.queue_data = Queue(maxsize=qsize)
        if nepochs is None:
            nepochs = 1
            self.loop_data = True
        else:
            self.loop_data = False

        if self._quat_regress:
            # self.total_items = nepochs * len(self._quat_viewpoint_list)
            self.total_items = nepochs * len(self._image_index)
        else:
            self.total_items = nepochs * len(self._image_index)
        self.queue_idx = Queue(maxsize=self.total_items)

        for nx in range(nepochs):
            if self._quat_regress:
                if self.shuffle:
                    # self._shuffle_quat_viewpoints_list()
                    self._shuffle_image_set_index()
                # for rx in range(len(self._quat_viewpoint_list)):
                for rx in range(len(self._image_index)):
                    self.queue_idx.put(rx)
            else:
                if self.shuffle:
                    self._shuffle_image_set_index()
                for rx in range(len(self._image_index)):
                    self.queue_idx.put(rx)

        # Debug dataset loading with these function calls
        # self.test_load_ycb_all()
        # self.test_load_quat_viewpoint()

        self.qthreads = []
        self.logger.info('Starting {:d} prefetch threads'.format(nthreads))
        for qx in range(nthreads):
            worker = Thread(target=self.fetch_data)
            worker.start()
            self.coord.register_thread(worker)
            self.qthreads.append(worker)


    def close_queue(self, e=None):
        self.logger.debug('Closing queue')
        self.coord.request_stop(e)
        try:
            while True:
                self.queue_idx.get(block=False)
        except Empty:
            self.logger.debug('Emptied idx queue')

        try:
            while True:
                self.queue_data.get(block=False)
        except Empty:
            self.logger.debug("Emptied data queue")

    def next_batch(self, batch_size, timeout=0.5):
        data = []
        cnt = 0
        while cnt < batch_size:
            try:
                dt = self.queue_data.get(timeout=timeout)
                self.total_items -= 1
                data.append(dt)
            except Empty:
                self.logger.debug('Example queue empty')
                if self.total_items <= 0 and not self.loop_data:
                    # Exhausted all data
                    self.close_queue()
                    break
                else:
                    continue
            cnt += 1
        if len(data) == 0:
            return
        batch_data = {}
        if self._quat_regress:
            items = self.all_items_quat_viewpoints
        else:
            items = self.all_items
        for k in items:
            try:
                batch_data[k] = []
                for dt in data:
                    batch_data[k].append(dt[k])
                batched = np.stack(batch_data[k])
                batch_data[k] = batched
            except Exception as e:
                self.logger.debug('Error with item: {}'.format(k))
                raise
        return batch_data

    def reset(self):
        np.random.seed(self.rng)
