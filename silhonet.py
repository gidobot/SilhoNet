from utils import pretty_line


class SilhoNet(object):
    def __init__(self,
                 im_bs,
                 mdl_im_bs,
                 im_h,
                 im_w,
                 mdl_im_h,
                 mdl_im_w,
                 roi_size,
                 num_classes,
                 mode='train',
                 norm="IN"):
        self.bs = im_bs
        self.mdl_im_batch = mdl_im_bs
        self.im_h = im_h
        self.im_w = im_w
        self.mdl_im_h = mdl_im_h
        self.mdl_im_w = mdl_im_w
        self.roi_size = roi_size
        self.mode = mode
        self.norm = norm
        self.num_classes = num_classes
        self.training = (self.mode == 'train')

    @property
    def mdl_im_tensor_shape(self):
        return [self.bs, self.mdl_im_batch, self.mdl_im_h, self.mdl_im_w, 3]

    @property
    def input_im_tensor_shape(self):
        return [self.bs, self.im_h, self.im_w, 3]

    @property
    def label_tensor_shape(self):
        return [self.bs, self.im_h, self.im_w, 1]

    @property
    def roi_tensor_shape(self):
        return [self.bs, 4]

    @property
    def class_tensor_shape(self):
        return [self.bs, 1]

    @property
    def quat_tensor_shape(self):
        return [self.bs, self.mdl_im_batch, 4]

    @property
    def batch_size(self):
        return self.bs

    @property
    def roi_shape(self):
        return [self.bs, self.roi_size, self.roi_size]

    @property
    def class_shape(self):
        return (self.bs,)

    @property
    def pred_quat_shape(self):
        return [self.bs, 4]

    @property
    def gt_quat_shape(self):
        return [self.bs, 4]

    @property
    def total_mdl_ims_per_batch(self):
        return self.bs * self.mdl_im_batch

    def print_net(self):
        if hasattr(self, 'mdl_im_net'):
            print('\n')
            pretty_line('Model Image Encoder')
            for k, v in sorted(self.mdl_im_net.items()):
                print(k + '\t' + str(v.get_shape().as_list()))

        if hasattr(self, 'im_net'):
            print('\n')
            pretty_line('Image Encoder')
            for k, v in sorted(self.im_net.items()):
                print(k + '\t' + str(v.get_shape().as_list()))

        if hasattr(self, 'mdl_resize_net'):
            print('\n')
            pretty_line('Model Feature Map Resize')
            for k, v in sorted(self.mdl_resize_net.items()):
                print(k + '\t' + str(v.get_shape().as_list()))

        if hasattr(self, 'seg_net'):
            print('\n')
            pretty_line('Silhouette Prediction Net')
            for k, v in sorted(self.seg_net.items()):
                print(k + '\t' + str(v.get_shape().as_list()))

        if hasattr(self, 'quat_net'):
            print('\n')
            pretty_line('3D Pose Prediction Net')
            for k, v in sorted(self.quat_net.items()):
                print(k + '\t' + str(v.get_shape().as_list()))

        return
