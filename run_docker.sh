nvidia-docker run -it \
  -v /home/gidobot/mnt/workspace/neural_networks/tensorflow/SilhoNet:/SilhoNet \
  -v /home/gidobot/mnt/storage:/home/gidobot/mnt/storage \
  -w /SilhoNet \
  tensorflow/tensorflow:silhonet \
  bash
