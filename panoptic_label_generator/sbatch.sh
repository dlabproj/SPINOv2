#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate spino
cd /work/dlclarge2/goenencd-denizgonenc/SPINOv2/panoptic_label_generator
python semantic_fine_tuning.py fit --trainer.devices [0] --config configs/semantic_cityscapes.yaml > logs/semantic_cityscapes.txt 2>&1
python boundary_fine_tuning.py fit --trainer.devices [0] --config configs/boundary_cityscapes.yaml > logs/boundary_cityscapes.txt 2>&1
python instance_clustering.py test --trainer.devices [0] --config configs/instance_cityscapes.yaml > logs/instance_cityscapes.txt 2>&1
#python -m panoptic_segmentation_model.scripts_dev.evaluate_labels --dataset_name cityscapes --gpu_id 0 /work/dlclarge2/guptaay-DL2025_RL1/SPINOv2/cityscapes /work/dlclarge2/goenencd-denizgonenc/SPINOv2/cityscapes_pseudolabels/
