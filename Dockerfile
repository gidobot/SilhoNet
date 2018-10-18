FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update && \
    apt-get install -y python3-tk tmux git libsm6

RUN pip install tqdm
RUN pip install scikit-image
RUN pip install transforms3d
RUN pip install tabulate

# Tensorpack used for Faster-RCNN ROI detection
# RUN pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
# RUN cd /home/gidobot/mnt/workspace/neural_networks/tensorflow/tensorpack && \
# 	python setup.py build && python setup.py install

RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
RUN pip install opencv-python