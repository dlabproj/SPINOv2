#!/bin/bash
source /work/dlclarge2/goenencd-denizgonenc/miniconda3/etc/profile.d/conda.sh
conda activate spino
cd /work/dlclarge2/goenencd-denizgonenc/SPINOv2/panoptic_label_generator
python instance_clustering.py test --trainer.devices [0] --config configs/instance_cityscapes.yaml > logs/instance_cityscapes.txt 2>&1
