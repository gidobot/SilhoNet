# SilhoNet
This is the Tensorflow implementation of SilhoNet from the paper "SilhoNet: An RGB Method for 3D Object Pose Estimation and Grasp Planning", submitted to ICRA 2018. The code supports training, validation, and testing for both the silhouette prediction and 3D pose estimation stages of the network on the YCB-Video dataset.

**SilhoNet: An RGB Method for 3D Object Pose Estimation and Grasp Planning**  
[Gideon Billings](https://people.eecs.berkeley.edu/~akar/), [Matthew Johnson-Roberson](https://people.eecs.berkeley.edu/~chaene/)  
ICRA 2018  
[**[arxiv]**](https://arxiv.org/abs/1809.06893)

## Setup
### Prerequisites
 - Linux or OSX (Tested on Ubuntu 14.04 and 16.04)
 - NVIDIA GPU + CUDA + CuDNN (CPU mode and CUDA without CuDNN may work with minimal modification, but untested)

### Prepare data
#### Requirements for testing
We assume you have cloned this repo, and the root directory is `$SilhoNet_ROOT`.

The network requires the YCB-Video dataset, augmented with groundtruth silhouette renderings, and renderings of the object models.

The YCB-Video datset can be downloaded from the official project site:  
[**[YCB-Video dataset]**](https://rse-lab.cs.washington.edu/projects/posecnn/)

Create a simlink to the root directory of the YCB dataset.
```
cd $SilhoNet_ROOT
ln -s $YCB_DIR data/YCB
```
where '$YCB_DIR' is the root directory of the YCB dataset.

The dataset_toolbox folder provides MATLAB scripts for generating the silhouette annotation files.  
**WARNING**: These scripts can take several days to run, depending on how much processing power is available. It is recommended to run them on a CPU server.

Run the script for generating the augmented annotations.

```
cd $SilhoNet_ROOT/dataset_toolbox
matlab -nodesktop -nosplash -r gen_full_silhouettes
```

For testing SilhoNet on predicted ROIs, we provide our Faster-RCNN detections file for the keyframe image set. This file should be downloaded to the `$SilhoNet_ROOT/data` folder.  
[**[Faster-RCNN detections]**](https://drive.google.com/file/d/18L1kYnZ57v8boyo9vsqhQfjpPjRkNtWM/view?usp=sharing)


#### Additional requirements for training
For training, the network also requires the COCO-2017 training images set, which can be downloaded form the COCO dataset site:  
[**[COCO dataset]**](http://cocodataset.org/#download)

Create a simlink to the COCO images directory
```
cd $SilhoNet_ROOT
ln -s $COCO_DIR/images data/COCO
```
where '$COCO_DIR' is the root directory of the COCO dataset.

Generate annotations for the synthetic data
```
cd $SilhoNet_ROOT/dataset_toolbox
matlab -nodesktop -nosplash -r gen_full_silhouettes
```

Generate the synthetic bounding box annotation files:
```
matlab -nodesktop -nosplash -r gen_bboxes_synthetic
```

Generate the `trainsyn.txt` image set file, which includes the supplementary synthetic images
```
sh gen_synthetic_image_sets.sh
```

### Setup nvidia-docker
We recommend using the provided docker image to run experiments without modifying your local system setup. These instructions assume you have installed docker with the nvidia-docker wrapper. The SilhoNet code base is mounted at runtime outside of the docker image for ease of development.

Build the docker image
```
cd $SilhoNet_ROOT
sudo docker build -t tensorflow/tensorflow:silhonet .
```

We have provided a `run_docker.sh` script for launching of the docker image. This script should be modified for your system.
1. Replace `/home/gidobot/mnt/workspace/neural_networks/tensorflow/SilhoNet` with the path to your SilhoNet directory, `$SilhoNet_ROOT`.
2. Replace `/home/gidobot/mnt/storage` with the storage folder containing the downloaded YCB and COCO datasets. The mounted path and original path to this folder must match for the simlinks to work.

### Pretrained models

We have provided configuration files with the parameters for replicating the test results reported in the paper. These can be loaded on runtime with the `--argsjs` parameter. The configuration files load our trained model weights which can be downloaded from the link below.  
[**[SilhoNet pretrained weights]**](https://drive.google.com/open?id=1vQtaokm8veDiLJpc6qfcJJNU6IXSx2aF)  
Extract the weights file under the `data` folder by running
```
tar -xvfz pretrained_weights.tar.gz -C $SilhoNet_ROOT/data/
```

## Testing

The runtime parameters for SilhoNet can be listed by running
```
python -m scripts.run_silhonet --help
```

### Silhouette prediction
Use the following command to test the silhouette prediction network:
```
python -m scripts.run_silhonet --mode test-seg --argsjs args/args_silhouette_test.json
```

By default, the test runs with the YCB dataset ground truth ROIs. To test with the Faster-RCNN predicted ROIs, set `use_pred_rois` to true in the config file.

The network saves the test results to `$logdir/table.txt`, where `logdir` is specified in the config file. The columns of the accuracy tables correspond to the threshold values, specified by the `eval_thresh` parameter, where the threshold value is used to convert the probability masks into binary masks. The accuracy values are IoU percentage scores.

### 3D pose prediction
Use the following command to test the full 3D pose prediction network:
```
python -m scripts.run_silhonet --mode test-quat --argsjs args/args_pose_test.json
```

By default, the test runs with the YCB dataset ground truth ROIs. To test with the Faster-RCNN predicted ROIs, set `use_pred_rois` to true in the config file.

The network saves the test results to `$logdir/table.txt` and `$logdir/angle_errors.mat`, where `logdir` is specified in the config file. The columns of the accuracy table corresponds to the angle error threshold values, where the accuracy values are the percentage of predicted poses that have an angle error less than the threshold. The angle_errors.mat file is used to plot accuracy against the PoseCNN published results.

To compare accuracy against PoseCNN, run the test with ground truth ROIs and copy the generated `angle_errors.mat` file to `dataset_toolbox/results_SilhoNet/angle_errors_gt.mat`. Then run the test with predicted ROIs and copy the generated `angle_errors.mat` file to `dataset_toolbox/results_SilhoNet/angle_errors_pred.mat`. There is a MATLAB script to run the evaluation
```
matlab -nodesktop -nosplash -r plot_accuracy_keyframe
```
The plots of the results are saved under the subdirectory `plots`.

### Visualizing test results in Tensorboard
Some test results are summerized to a tensorboard event file under the specified `logdir` directory. These visualizations can be helpful for debugging and can be viewed in a web browser by running
```
tensorboard --logdir $logdir --port $PORT
```
**NOTE**: If running tensorboard in docker, you will need to forward the port out of docker to view in your local browser. If logging the results to a location accessible outside of docker (recommended), you can run tensorboard on your local system.

## Training

The runtime parameters for SilhoNet can be listed by running
```
python -m scripts.run_silhonet --help
```

### Silhouette prediction
We provide imagenet pretrained weights for the VGG16 backbone network which can be downloaded from the link below.  
[**[VGG16 ImageNet pretrained weights]**](https://drive.google.com/open?id=1mWpx8oaw__GPnxDanNtdw8o-MIf_6dcH)  
Extract the weights file under the `data` folder by running
```
tar -xvfz imagenet_weights.tar.gz -C $SilhoNet_ROOT/data/
```

Use the following command to train the silhouette prediction network with the default parameters:
```
python -m scripts.run_silhonet --mode train-seg --argsjs args/args_silhouette_train.json
```
**NOTE**: In the release code, it is expected that the silhouette prediction stage is trained before the 3D pose prediction stage, as the network weights for the silhouette prediction stage are loaded for both training and testing the 3D pose prediction stage. 

Training checkpoints are saved to the `logdir` directory specified in the config file.

### 3D pose prediction
Use the following command to train the 3D pose prediction network with the default parameters:
```
python -m scripts.run_silhonet --mode train-quat --argsjs args/args_pose_train.json
```

Training checkpoints are saved to the `logdir` directory specified in the config file.

### Visualizing training with Tensorboard
Training is summerized to a tensorboard event file under the specified `logdir` directory.  Visualize trianing in a web browser by running
```
tensorboard --logdir $logdir --port $PORT
```
**NOTE**: If running tensorboard in docker, you will need to forward the port out of docker to view in your local browser. If logging the results to a location outside of docker (recommended), you can run tensorboard on your local system.

## Citation
If you use our code, we request you to cite the following work.
```
@ARTICLE{2018arXiv180906893B,
   author = {{Billings}, G. and {Johnson-Roberson}, M.},
    title = "{SilhoNet: An RGB Method for 3D Object Pose Estimation and Grasp Planning}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1809.06893},
 primaryClass = "cs.CV",
 keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Robotics},
     year = 2018,
    month = sep,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180906893B},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
