import numpy as np
import json
import os
import skimage
from scipy.io import loadmat
from skimage.io import imread
from skimage.transform import resize
from transforms3d.quaternions import quat2mat, mat2quat
from transforms3d.euler import euler2mat, mat2euler
from math import atan2
import random

def rgb_to_hsv(rgb):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    rgb = rgb.astype('float')
    hsv = np.zeros_like(rgb)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = rgb[..., 3:]
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb[..., :3], axis=-1)
    minc = np.min(rgb[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv

def hsv_to_rgb(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    rgb = np.empty_like(hsv)
    rgb[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
    rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return rgb.astype('uint8')


def shift_hsv(img,hf,sf,vf):
    hsv = skimage.color.rgb2hsv(img)
    # hsv = rgb_to_hsv(img)
    hsv[...,0]=np.clip(hsv[...,0]*hf,0.0,1.0)
    hsv[...,1]=np.clip(hsv[...,1]*sf,0.0,1.0)
    hsv[...,2]=np.clip((hsv[...,2]*vf),0.0,1.0)
    rgb = skimage.img_as_ubyte(skimage.color.hsv2rgb(hsv))
    # rgb=hsv_to_rgb(hsv)
    return rgb


def read_im(f, train=False, bg_file=None):
    im = skimage.img_as_float(imread(f))
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=2)
    if im.shape[-1] == 4: # assume only synthetic images have 4 channels
        # Hacky background insertion from coco dataset for now
        if bg_file is not None:
            bg_im = skimage.img_as_float(imread(bg_file))
            if bg_im.ndim == 2:
                bg_im = np.stack([bg_im, bg_im, bg_im], axis=2)
            aspect = im.shape[0]/im.shape[1]
            bg_aspect = bg_im.shape[0]/bg_im.shape[1]
            if bg_aspect > aspect:
                max_ind = int(bg_im.shape[1]*aspect)
                bg_im = bg_im[0:max_ind,...]
            else:
                max_ind = int(bg_im.shape[0]/aspect)
                bg_im = bg_im[:,0:max_ind,...]
            bg_im = resize(bg_im, (im.shape[0], im.shape[1]))
            alpha = np.expand_dims(im[..., 3], 2)
            im = im[..., :3] * alpha + bg_im[...]*(1 - alpha)
        else:
            alpha = np.expand_dims(im[..., 3], 2)
            im = im[..., :3] * alpha + (1 - alpha)
    if train:
        im = skimage.img_as_ubyte(im)
        hf = random.uniform(1.0/1.5,1.5)
        sf = random.uniform(1.0/1.5,1.5)
        vf = random.uniform(1.0/1.5,1.5)
        im = shift_hsv(im,hf,sf,vf)
        im = skimage.img_as_float(im)
    return im


def read_depth(f):
    im = skimage.img_as_float(imread(f)) * 10
    if im.ndim == 2:
        im = np.expand_dims(im, 2)
    return im

def read_label(f):
    im = imread(f)
    #if im.ndim == 2:
    #    im = np.expand_dims(im, 2)
    return im


def load_pred_rois(f):
    json_data = open(f).read()
    pred_dict = json.loads(json_data)
    return pred_dict

# Return: list [cls_idx, qw, qx, qy, qz, tx, ty, tz] with dim [None, 8]
def read_poses(f):
    meta_data = loadmat(f)
    meta_data['cls_indexes'] = meta_data['cls_indexes'].flatten()
    poses = meta_data['poses']
    if len(poses.shape) == 2:
        poses = np.reshape(poses, (3, 4, 1))
    num = poses.shape[2]
    qt = np.zeros((num, 8), dtype=np.float32)
    for j in range(num):
        R = poses[:, :3, j]
        T = poses[:, 3, j]

        # use translation to get apparant orientation of object in the
        # camera view.
        pitch_angle = atan2(T[0],T[2]);
        roll_angle = -atan2(T[1],T[2]);
        eul = [roll_angle, pitch_angle, 0];
        rot = euler2mat(roll_angle, pitch_angle, 0, axes='rxyz')
        R = np.dot(np.linalg.inv(rot), R)

        qt[j, 0]   = meta_data['cls_indexes'][j]
        qt[j, 1:5] = mat2quat(R) # (w,x,y,z)
        qt[j, 5:]  = T
    return qt

def read_segments(f):
    """Reads projected segmentation masks from file
    Args:
        f (string): Path to the .mat file

    Returns:
        dict: A dictionary of segmentation masks, indexed by the class id number.
    """
    segment_data = loadmat(f)
    cls_idxs = segment_data['mask']['cls_indexes'][0][0][0]
    segments = segment_data['mask']['segments_crop'][0][0]
    segment_dict = {}
    for i in range(len(cls_idxs)):
        idx = cls_idxs[i]
        segment_dict[idx] = segments[i]
    return segment_dict

def read_quat_viewpoint(f):
    """Reads projected silhouette viewpoint mask with associated quaternion from file
    Args:
        f (string): Path to the .mat file

    Returns:
        np array, np array: A matrix for the mask. A list for the quaternion
    """
    data = loadmat(f)
    mask = data['viewpoint']['mask'][0,0]
    quat = data['viewpoint']['quat'][0,0][0]
    return mask, quat

def read_viewpoint_ids(f):
    """Reads viewpoint ids from file
    Args:
        f (string): Path to the .mat file

    Returns:
        dict: A dictionary of viewpoint ids, indexed by the class id number.
    """
    segment_data = loadmat(f)
    cls_idxs = segment_data['mask']['cls_indexes'][0][0][0]
    ids = segment_data['mask']['viewpoints'][0][0][0]
    id_dict = {}
    for i in range(len(cls_idxs)):
        idx = cls_idxs[i]
        id_dict[idx] = ids[i] - 1 # convert from matlab index 1
    return id_dict

def read_viewpoints(f):
    """Reads viewpoint silhouette masks from file
    Args:
        f (string): Path to the .mat file

    Returns:
        dict: A dictionary of viewpoints, indexed by the viewpoint id number.
    """
    vp_data = loadmat(f)
    vps = vp_data['viewpoints']['masks'][0][0]
    vp_dict = {}
    for i in range(vps.shape[0]):
        vp_dict[i] = vps[i]
    return vp_dict

def read_viewpoint_yaws(f):
    """Reads viewpoint yaws from file
    Args:
        f (string): Path to the .mat file

    Returns:
        dict: A dictionary of viewpoint yaws, indexed by the class id number.
    """
    segment_data = loadmat(f)
    cls_idxs = segment_data['mask']['cls_indexes'][0][0][0]
    yaws = segment_data['mask']['viewpoint_yaw'][0][0][0]
    yaw_dict = {}
    for i in range(len(cls_idxs)):
        idx = cls_idxs[i]
        yaw_dict[idx] = yaws[i]
    return yaw_dict

def read_viewpoint_quats(f):
    """Reads viewpoint qauternion from file
    Args:
        f (string): Path to the .mat file

    Returns:
        dict: A dictionary of viewpoint quaternions, indexed by the class id number.
    """
    vp_data = loadmat(f)
    vpq = vp_data['viewpoints']['quaternion'][0][0]
    quat_dict = {}
    for i in range(vpq.shape[0]):
        quat_dict[i] = vpq[i]
    return quat_dict

def read_occluded_segments(f):
    """Reads projected occluded segmentation masks from file
    Args:
        f (string): Path to the .mat file

    Returns:
        dict: A dictionary of segmentation masks, indexed by the class id number.
    """
    segment_data = loadmat(f)
    cls_idxs = segment_data['mask']['cls_indexes'][0][0][0]
    segments = segment_data['mask']['occluded_segments_crop'][0][0]
    segment_dict = {}
    for i in range(len(cls_idxs)):
        idx = cls_idxs[i]
        segment_dict[idx] = segments[i]
    return segment_dict

def read_roi(f):
    with open(f) as fd:
        roi_line = fd.readlines()
    roi_line = [x.strip() for x in roi_line if x.strip()]
    roi_dict = {}
    for line in roi_line:
        roi_list = line.split()
        # roi coordinates (y1, x1, y2, x2)
        roi_coords = [roi_list[2], roi_list[1], roi_list[4], roi_list[3]]
        roi_dict[roi_list[0]] = [float(x) for x in roi_coords]
    return roi_dict

def read_classes(f):
    mat = loadmat(f)
    classes = [int(x) for x in mat['cls_indexes'].ravel()]
    return classes

def read_camera(f):
    cam = loadmat(f)
    Rt = cam['extrinsic'][:3]
    K = cam['K']
    return K, Rt

def read_quat(f):
    cam = loadmat(f)
    q = cam['quat'].ravel()
    return q

def read_vol(f, tsdf=False):
    def get_vox(f):
        try:
            data = loadmat(f, squeeze_me=True)
        except:
            print('Error reading {:s}'.format(f))
            return None
        vol = np.transpose(data['Volume'].astype(np.bool), [0, 2, 1])
        vol = vol[:, ::-1, :]
        return vol

    def get_tsdf(f, trunc=0.2):
        try:
            data = loadmat(f, squeeze_me=True)
        except:
            print('Error reading {:s}'.format(f))
            return None

        tsdf = data['tsdf']
        tsdf[tsdf < -trunc] = -trunc
        tsdf[tsdf > trunc] = trunc
        tsdf = np.transpose(tsdf, [0, 2, 1])
        tsdf = tsdf[:, ::-1, :]
        return tsdf

    load_func = get_tsdf if tsdf else get_vox
    vol = load_func(f).astype(np.float32)
    vol = vol[..., np.newaxis]
    return vol


def pad_batch(batch_data, bs):
    for k, v in batch_data.items():
        n = v.shape[0]
        to_pad = bs - n
        if to_pad == 0:
            continue
        pad = np.stack([v[0, ...]] * to_pad, axis=0)
        batch_data[k] = np.concatenate([v, pad], axis=0)
    return batch_data


def subsample_grid(grids, sub_ratio):
    def subsample(g):
        ss = np.array(g.shape) / sub_ratio
        sub_grid = np.zeros(ss, dtype=np.bool)
        for ix in range(sub_ratio):
            for jx in range(sub_ratio):
                for kx in range(sub_ratio):
                    sub_grid = np.logical_or(
                        sub_grid,
                        g[ix::sub_ratio, jx::sub_ratio, kx::sub_ratio])
        return sub_grid
    return [subsample(g) for g in grids]