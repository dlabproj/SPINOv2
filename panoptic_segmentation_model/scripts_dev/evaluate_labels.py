# pylint: disable=unbalanced-tuple-unpacking

import sys
from pathlib import Path

# Run from root of SPINO folder
sys.path.append(Path(__file__).parent.parent.absolute().as_posix())
sys.path.append(Path(__file__).parent.parent.parent.absolute().as_posix())

import argparse
import socket

import torch
from algos import BinaryMaskLoss, CenterLoss, InstanceSegAlgo, OffsetLoss
from datasets import Cityscapes
from eval import PanopticEvaluator, SemanticEvaluator
from io_utils.visualizations import plot_confusion_matrix
from misc import train_utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from yacs.config import CfgNode as CN

print(f"Running on host: {socket.gethostname()}")

parser = argparse.ArgumentParser(description="Evaluate labels")
parser.add_argument("gt_path", help="Path to ground truth labels")
parser.add_argument("pred_path", help="Path to predicted labels")
parser.add_argument("--dataset_name", required=True, help="Name of dataset")
parser.add_argument("--gpu_id", default="0", help="GPU to use")
args = parser.parse_args()

print(f"GT: {args.gt_path}\nPRED: {args.pred_path}\n")

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

dataset_cfg = CN()
dataset_cfg.name = args.dataset_name
dataset_cfg.center_heatmap_sigma = 8
dataset_cfg.return_only_rgb = False
dataset_cfg.small_instance_weight = 3
dataset_cfg.remove_classes = []
dataset_cfg.augmentation = CN()
dataset_cfg.augmentation.active = False
dataset_cfg.normalization = CN()
dataset_cfg.normalization.active = False

#subset_indices = list(range(0, 500, 10))

# Create dataloader for ground truth and predicted labels
# if args.dataset_name == "kitti_360":
#     dataset_cfg.feed_img_size = [376, 1408]
#     dataset_cfg.small_instance_area_full_res = 1034
#     dataset_cfg.train_split = "sequence"
#     dataset_cfg.train_sequences = ["00", "02", "03", "04", "05", "06", "07", "09"]
#     dataset_cfg.val_split = "sequence"
#     dataset_cfg.val_sequences = ["10"]
#     dataset_cfg.sequence_reference_mode = "semantic"
#     dataset_cfg.label_mode = "kitti-360-14"
#
#     dataset_cfg.path = args.gt_path
#     gt_dataset = Kitti360(dataset_cfg.val_split, dataset_cfg, label_mode=dataset_cfg.label_mode, sequences=dataset_cfg.val_sequences,
#                           sequence_reference_mode=dataset_cfg.sequence_reference_mode)
#     dataset_cfg.path = args.pred_path
#     pred_dataset = Kitti360(dataset_cfg.val_split, dataset_cfg, label_mode=dataset_cfg.label_mode, mode_path="_plabels",
#                             is_gt=False, sequences=dataset_cfg.val_sequences,
#                             sequence_reference_mode=dataset_cfg.sequence_reference_mode)

if args.dataset_name == "cityscapes":
    dataset_cfg.feed_img_size = [1024, 2048]
    dataset_cfg.small_instance_area_full_res = 4096
    dataset_cfg.train_split = "train"
    dataset_cfg.val_split = "val"
    dataset_cfg.label_mode = "cityscapes-19"

    mode = "val"
    subset_indices = None
    # mode = "train"
    # subset_indices = [12, 324, 450, 608, 742, 768, 798, 836, 1300, 2892]
    # subset_indices = sorted([71, 2896, 340, 771, 423, 1497, 2880, 524, 654, 765, 589, 734, 84, 652, 466, 421, 2893, 982, 106, 189, 5, 766, 339, 685, 1628, 976, 380, 783, 697, 772, 1395, 2879, 573, 1635, 1246, 1331, 1035, 653, 2914, 729, 565, 28, 1523, 70, 632, 385, 773, 970, 1472, 755, 763, 1105, 1286, 1273, 523, 664, 703, 1412, 10, 540, 505, 170, 511, 453, 543, 849, 276, 1675, 1378, 575, 1578, 1485, 117, 2689, 2209, 41, 1182, 1890, 1183, 2690, 388, 527, 1344, 172, 718, 180, 1102, 198, 538, 1504, 138, 73, 16, 1038, 1196, 211, 1021, 774, 179, 408])
    # subset_indices = [5, 11, 16, 28, 42, 90, 106, 107, 111, 162, 254, 323, 326, 327, 339, 340, 376, 385, 386, 400, 410, 420, 423, 425, 451, 466, 516, 524, 525, 527, 529, 539, 540, 555, 565, 628, 631, 650, 652, 654, 657, 689, 690, 700, 708, 710, 719, 720, 721, 725, 728, 729, 734, 738, 745, 746, 750, 755, 767, 769, 772, 783, 790, 793, 808, 890, 904, 970, 1039, 1238, 1239, 1299, 1301, 1302, 1303, 1304, 1339, 1368, 1376, 1409, 1512, 1530, 1535, 1820, 1900, 1914, 1992, 2209, 2445, 2446, 2462, 2548, 2653, 2685, 2854, 2856, 2865, 2891, 2893, 2896]
    # subset_indices = [11, 90, 106, 107, 162, 326, 339, 376, 386, 400, 420, 423, 425, 451, 524, 525, 527, 529, 539, 555, 628, 631, 654, 689, 721, 728, 729, 734, 746, 755, 767, 769, 772, 793, 1039, 1299, 1301, 1302, 1339, 1376, 1535, 1820, 1900, 1914, 2445, 2685, 2856, 2891, 2893, 2896]
    # subset_indices = sorted([2743, 1670, 2069, 295, 312, 2723, 1773, 1895, 2026, 114, 164, 2115, 24, 812, 2748, 2744, 305, 2803, 253, 1683, 1601, 2554, 1982, 1682, 2727, 2742, 2845, 160, 2746, 2068, 1681, 2086, 158, 1382, 82, 1129, 22, 296, 343, 94, 2799, 2650, 2739, 2924, 2234, 2073, 2067, 1471, 2485, 2652, 2590, 2794, 2710, 244, 2747, 313, 95, 2751, 2925, 2066, 2539, 2576, 1749, 2644, 294, 2729, 2535, 2724, 1459, 2805, 2737, 2063, 1092, 2728, 2806, 285, 1633, 2507, 2608, 1554, 2386, 2730, 2065, 2064, 2732, 2626, 2430, 2407, 2054, 2752, 2801, 293, 307, 2232, 2538, 2377, 152, 2478, 1112, 2726])

    dataset_cfg.path = args.gt_path
    gt_dataset = Cityscapes(mode, dataset_cfg, label_mode=dataset_cfg.label_mode, subset=subset_indices)
    dataset_cfg.path = args.pred_path
    pred_dataset = Cityscapes(mode, dataset_cfg, label_mode=dataset_cfg.label_mode,
                              mode_path=mode, is_gt=False)  # val_plabels
    assert len(gt_dataset) == len(pred_dataset)

# elif args.dataset_name == "pascal_voc":
#     dataset_cfg.feed_img_size = [512, 512]
#     dataset_cfg.small_instance_area_full_res = 1034
#     dataset_cfg.train_split = "train"
#     dataset_cfg.val_split = "val"
#     dataset_cfg.label_mode = "pascal_voc"
#
#     dataset_cfg.path = args.gt_path
#     gt_dataset = PascalVOC("val", dataset_cfg, label_mode=dataset_cfg.label_mode)
#     dataset_cfg.path = args.pred_path
#     pred_dataset = PascalVOC("val", dataset_cfg, label_mode=dataset_cfg.label_mode)
#     assert len(gt_dataset) == len(pred_dataset)
#
# elif args.dataset_name == "phenobench":
#     dataset_cfg.feed_img_size = [1024, 1024]
#     dataset_cfg.small_instance_area_full_res = 2048
#     dataset_cfg.train_split = "train"
#     dataset_cfg.val_split = "val"
#     dataset_cfg.label_mode = "phenobench"
#
#     dataset_cfg.path = args.gt_path
#     gt_dataset = PhenoBench("val", dataset_cfg, label_mode=dataset_cfg.label_mode)
#     dataset_cfg.path = args.pred_path
#     pred_dataset = PhenoBench("val", dataset_cfg, label_mode=dataset_cfg.label_mode)
#     assert len(gt_dataset) == len(pred_dataset)

else:
    raise NotImplementedError(f"Dataset {args.dataset_name} not implemented")

gt_dataloader = DataLoader(gt_dataset, batch_size=8, shuffle=False, num_workers=8)
pred_dataloader = DataLoader(pred_dataset, batch_size=8, shuffle=False, num_workers=8)

# Create algorithms and evaluators
instance_center_loss = CenterLoss()
instance_offset_loss = OffsetLoss()
binary_mask_loss = BinaryMaskLoss()
panoptic_eval = PanopticEvaluator(stuff_list=gt_dataset.stuff_classes,
                                  thing_list=gt_dataset.thing_classes,
                                  label_divisor=1000, void_label=-1)
instance_algo = InstanceSegAlgo(instance_center_loss, instance_offset_loss, panoptic_eval,
                                binary_mask_loss)
num_classes = len(gt_dataset.stuff_classes) + len(gt_dataset.thing_classes)
semantic_eval = SemanticEvaluator(num_classes=num_classes, ignore_classes=gt_dataset.ignore_classes)

# Iterate over the dataloaders to compute the metrics
confusion_matrix = torch.zeros((semantic_eval.num_classes, semantic_eval.num_classes),
                               dtype=torch.int64, device=device)
for gt_item, pred_item in tqdm(zip(gt_dataloader, pred_dataloader), total=len(gt_dataloader),
                               desc="Evaluate"):
    gt_item = train_utils.dict_to_cuda(gt_item, device)
    pred_item = train_utils.dict_to_cuda(pred_item, device)

    # Panoptic segmentation
    gt_item["panoptic"], _ = instance_algo.panoptic_fusion(gt_item["semantic"], gt_item["center"],
                                                           gt_item["offset"])
    pred_item["panoptic"], _ = instance_algo.panoptic_fusion(pred_item["semantic"],
                                                             pred_item["center"],
                                                             pred_item["offset"])
    panoptic_eval.update(gt_item["panoptic"], pred_item["panoptic"])

    # Semantic segmentation
    confusion_matrix += semantic_eval.compute_confusion_matrix(pred_item["semantic"],
                                                               gt_item["semantic"])

# Classes that are not covered in the ground truth should not be considered
indices_with_gt = confusion_matrix.sum(dim=1) != 0
sem_miou_score = semantic_eval.compute_sem_miou(confusion_matrix)[indices_with_gt].mean()
acc_score = semantic_eval.compute_sem_miou(confusion_matrix, sum_pixels=True)

print("******************")
print("EVALUATION RESULTS")
print("******************")
print(panoptic_eval.evaluate())
print(f"Accuracy: {acc_score}")
print(f"Semantic mIoU: {sem_miou_score}")
print(f"Confusion matrix:\n{confusion_matrix}")
print("******************")

# Plot and save confusion matrix
conf_mat_plt = plot_confusion_matrix(confusion_matrix, [], label_mode=dataset_cfg.label_mode)
conf_mat_plt.figure.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
