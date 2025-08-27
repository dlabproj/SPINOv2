# Panoptic Label Generator: Complete Code Documentation and Execution Flow

This document provides a comprehensive line-by-line explanation of the panoptic label generator components in SPINOv2, focusing on the semantic and boundary detection modules and their execution flow.

## 🏗️ Architecture Overview

The panoptic label generator consists of four main components:

1. **FineTuner** (`fine_tuning.py`) - Base class for DINOv2 feature extraction
2. **SemanticFineTuner** (`semantic_fine_tuning.py`) - Semantic segmentation module  
3. **BoundaryFineTuner** (`boundary_fine_tuning.py`) - Boundary detection module
4. **InstanceCluster** (`instance_clustering.py`) - Panoptic fusion module

## 📊 Execution Flow

```
Input RGB Image (B, 3, H, W)
        ↓
    DINOv2 Encoder (frozen)
        ↓
    Feature Maps (B, feat_dim, H_patch, W_patch)
        ↓
┌─────────────────┬─────────────────┐
│  Semantic Head  │  Boundary Head  │
│                 │                 │
│ Semantic Logits │ Boundary Probs  │
│ (B, C, H, W)   │ (B, 1, H, W)   │
└─────────────────┴─────────────────┘
        ↓                ↓
    Semantic Map     Boundary Map
        ↓                ↓
        └────────────────┘
                ↓
        Instance Clustering
                ↓
        Panoptic Segmentation
```

---

## 🧠 1. Base FineTuner Class (`fine_tuning.py`)

### Line-by-Line Analysis

```python
# Lines 1-14: Import statements
from typing import List, Optional
import pytorch_lightning as pl
import torch
from models.dino_v2 import (
    dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14,
)
from torch import nn
import torch.nn.functional as F
from models.dino_vit_adapter import ViTAdapter
```

**Explanation:**
- Imports PyTorch Lightning for training orchestration
- Imports DINOv2 model variants (Small, Base, Large, Giant with 14x14 patches)
- Imports ViTAdapter for additional feature extraction capabilities

```python
# Lines 17-29: Class definition and initialization
class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_adapter: bool = False):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.use_adapter = use_adapter
```

**Explanation:**
- **dinov2_vit_model**: Specifies which DINOv2 variant to use ('vits14', 'vitb14', 'vitl14', 'vitg14')
- **blocks**: List of transformer block indices to extract features from (if None, uses only last block)
- **upsample_factor**: Factor to upsample feature maps before head processing
- **use_adapter**: Whether to use ViT-Adapter for enhanced feature extraction

```python
# Lines 31-48: Model initialization based on variant
if self.use_adapter:
    self.encoder = ViTAdapter()
    print(f'[ENCODER] Using encoder: ViTAdapter')
elif dinov2_vit_model == 'vits14':
    self.encoder = dinov2_vits14(pretrained=True)
    print(f'[ENCODER] Using encoder: ViT-S14')
# ... similar for other variants
else:
    raise ValueError(f'Unknown model {dinov2_vit_model}')
```

**Explanation:**
- Loads pretrained DINOv2 models based on the specified variant
- ViT-S14: Smallest model (384 dim, ~22M params)
- ViT-B14: Base model (768 dim, ~86M params)  
- ViT-L14: Large model (1024 dim, ~300M params)
- ViT-G14: Giant model (1536 dim, ~1.1B params)

```python
# Lines 52-63: Encoder freezing and feature setup
if self.use_adapter == False:
    for param in self.encoder.parameters():
        param.requires_grad = False

self.feat_dim = self.encoder.num_features
self.patch_size = self.encoder.patch_size
self.encoder.mask_token = None
```

**Explanation:**
- **Freezing**: DINOv2 weights are frozen to preserve pretrained features
- **feat_dim**: Feature dimension (384/768/1024/1536 for S/B/L/G)
- **patch_size**: Always 14 for DINOv2 models (14x14 pixel patches)
- **mask_token**: Disabled to prevent DDP issues

### Forward Encoder Method

```python
# Lines 116-163: Feature extraction from DINOv2
def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
    img_h, img_w = img.shape[2:]
    patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size
```

**Key Concepts:**
- **Patch-based processing**: Images are divided into 14x14 patches
- **Feature maps**: Output spatial resolution is H/14 × W/14

```python
# ViT-Adapter path (Lines 124-134)
if self.use_adapter:
    f1, f2, f3, f4 = self.encoder.forward(img)
    _, _, h_f1, w_f1 = f1.shape
    
    f2_upsampled = F.interpolate(f2, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
    f3_upsampled = F.interpolate(f3, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
    f4_upsampled = F.interpolate(f4, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
    
    x = torch.cat([f1, f2_upsampled, f3_upsampled, f4_upsampled], dim=1)
    return x
```

**Explanation:**
- ViT-Adapter extracts multi-scale features (f1, f2, f3, f4)
- All features are upsampled to f1's resolution for concatenation
- Results in 4× channel dimensions (4 × feat_dim)

```python
# Standard DINOv2 path (Lines 137-163)
with torch.no_grad():
    block_outputs = self.encoder.forward_features(
        img, return_attention_features=return_attention_features,
        return_blocks=self.blocks)
    
    if self.blocks is None:
        block_outputs = [block_outputs]
    
    outs = []
    for x in block_outputs:
        x = x[feature_key]
        # Handle different feature types (x, q, k, v, attn)
        outs.append(x)
    
    x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * num_blocks)
    x = x[:, 1:, :]  # Remove CLS token (B, Patches, feat_dim)
    x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
    x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h, patches_w))
    
    if self.upsample_factor is not None:
        x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, 
                                    mode='bilinear', align_corners=False)
```

**Key Transformations:**
1. **Feature Extraction**: Get features from specified transformer blocks
2. **CLS Token Removal**: Remove class token (first token) to keep only spatial features
3. **Reshape**: Convert from sequence (B, Patches, C) to spatial (B, C, H, W)
4. **Upsampling**: Optional upsampling to increase spatial resolution

---

## 🎨 2. Semantic Fine-Tuner (`semantic_fine_tuning.py`)

### Imports and Setup

```python
# Lines 1-25: Imports and warning filters
import os, warnings
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn.functional as F
import torchvision.transforms as T
from fine_tuning import FineTuner
from PIL import Image
from pytorch_lightning.cli import LightningCLI
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from models.dino_vit_adapter import ViTAdapter

# Ignore interpolation warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
```

### Class Definition and Initialization

```python
# Lines 28-57: SemanticFineTuner class definition
class SemanticFineTuner(FineTuner):
    """Fine-tunes a small head on top of the DINOv2 model for semantic segmentation."""
```

**Parameters Explained:**
- **num_classes**: Number of semantic classes (e.g., 19 for Cityscapes)
- **train_output_size**: Spatial resolution during training (e.g., [512, 1024])
- **ignore_index**: Label value to ignore in loss computation (typically 255)
- **top_k_percent_pixels**: Fraction of hardest pixels for hard pixel mining
- **test_output_size**: Final output resolution during inference
- **test_multi_scales**: Scales for multi-scale test augmentation

```python
# Lines 59-87: Head architecture determination
def __init__(self, dinov2_vit_model: str, num_classes: int, train_output_size: Tuple[int, int],
             blocks: Optional[List[int]] = None, upsample_factor: Optional[float] = None,
             head: str = 'linear', ignore_index: int = -100, top_k_percent_pixels: float = 1.0,
             test_output_size: Optional[Tuple[int, int]] = None,
             test_multi_scales: Optional[List[int]] = None,
             test_plot: bool = False, test_save_dir: Optional[str] = None,
             use_adapter: bool = False):
    
    super().__init__(dinov2_vit_model=dinov2_vit_model, blocks=blocks,
                     upsample_factor=upsample_factor, use_adapter=use_adapter)
    
    # Store semantic-specific parameters
    self.num_classes = num_classes
    self.train_output_size = train_output_size
    self.ignore_index = ignore_index
    self.top_k_percent_pixels = top_k_percent_pixels
    self.test_output_size = test_output_size
    self.test_multi_scales = test_multi_scales
    self.test_plot = test_plot
    self.test_save_dir = test_save_dir
```

### Head Architecture Setup

```python
# Lines 82-117: Head input dimension calculation and architecture
if self.use_adapter:
    # For adapter, we use 4x feature dimension (multi-scale features)
    head_input_dim = self.feat_dim * self.num_blocks * 4
else:
    # For regular DINOv2, use feat_dim * num_blocks
    head_input_dim = self.feat_dim * self.num_blocks

if head == 'linear':
    # Simple 1x1 convolution for classification
    self.head = nn.Conv2d(head_input_dim, num_classes, kernel_size=1, stride=1, padding=0)
elif head == 'knn':
    # K-Nearest Neighbors classifier (non-parametric)
    self.head = KNeighborsClassifier(n_neighbors=5, leaf_size=10)
    self.knn_X = []  # Storage for training features
    self.knn_y = []  # Storage for training labels
elif head == 'cnn':
    # Multi-layer CNN head
    self.head = nn.Sequential(
        nn.Conv2d(head_input_dim, 300, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(300, 300, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(300, 200, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(200, num_classes, kernel_size=3, stride=1, padding=1),
    )
elif head == 'mlp':
    # Multi-layer perceptron with 1x1 convolutions
    self.head = nn.Sequential(
        nn.Conv2d(head_input_dim, 300, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(300, 300, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(300, 200, kernel_size=1, stride=1, padding=0),
        nn.ReLU(),
        nn.Conv2d(200, num_classes, kernel_size=1, stride=1, padding=0),
    )
```

**Head Architecture Analysis:**
- **Linear**: Simplest approach, direct mapping from features to classes
- **KNN**: Non-parametric, finds 5 nearest training features for classification
- **CNN**: Uses 3x3 convolutions to incorporate spatial context
- **MLP**: Uses 1x1 convolutions (point-wise processing)

### Forward Pass

```python
# Lines 119-136: Forward method
def forward(self, x: torch.Tensor) -> torch.Tensor:
    print("[DEBUG] Input to SemanticFineTuner.forward, x.shape:", x.shape)
    x = self.forward_encoder(x)  # Extract DINOv2 features (B, feat_dim, H, W)
    
    if isinstance(self.head, KNeighborsClassifier):
        if self.training:
            return x  # Return features during training for KNN storage
        # During inference, use stored KNN model
        feat_shape = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, feat_shape[1])  # (B*H*W, feat_dim)
        x = x.detach().cpu().numpy()
        x = self.head.predict_proba(x)  # (B*H*W, num_classes)
        x = torch.from_numpy(x).to(self.device)
        x = x.reshape(feat_shape[0], feat_shape[2], feat_shape[3], -1).permute(0, 3, 1, 2)
    else:
        x = self.head(x)  # Apply parametric head (B, num_classes, H, W)
    
    # Interpolate to training output size
    x = nn.functional.interpolate(x, size=self.train_output_size, mode='bilinear',
                                  align_corners=False)
    return x
```

**Forward Pass Flow:**
1. **Feature Extraction**: Get DINOv2 features via `forward_encoder`
2. **Head Processing**: Apply appropriate head architecture
3. **Spatial Interpolation**: Resize to target training resolution
4. **KNN Special Case**: Store features during training, predict during inference

### Training Step

```python
# Lines 138-173: Training step implementation
def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
    # Memory management for large models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    rgb = train_batch['rgb']  # Input images (B, 3, H, W)
    sem = train_batch['semantic'].long()  # Ground truth semantic labels (B, H, W)
    
    if isinstance(self.head, KNeighborsClassifier):
        # KNN: Store features and labels for later fitting
        x = self(rgb)  # Get features (B, feat_dim, H, W)
        feat_h, feat_w = x.shape[2:]
        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # (B*H*W, feat_dim)
        x = x.detach().cpu().numpy()
        self.knn_X.append(x)
        
        # Resize semantic labels to feature resolution
        sem = TF.resize(sem, [feat_h, feat_w], interpolation=InterpolationMode.NEAREST)
        sem = sem.reshape(-1).detach().cpu().numpy()
        self.knn_y.append(sem)
        
        loss = torch.tensor([0.0], requires_grad=True).to(self.device)  # Dummy loss
    else:
        # Parametric heads: Standard cross-entropy training
        sem = TF.resize(sem, self.train_output_size, interpolation=InterpolationMode.NEAREST)
        pred = self(rgb)  # Forward pass (B, num_classes, H, W)
        loss = F.cross_entropy(pred, sem, ignore_index=self.ignore_index, reduction='none')
        
        # Hard pixel mining: Focus on difficult pixels
        if self.top_k_percent_pixels < 1.0:
            loss = loss.contiguous().view(-1)
            top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
            loss, _ = torch.topk(loss, top_k_pixels)  # Keep only hardest pixels
        loss = loss.mean()
    
    self.log('train_loss', loss)
    return loss
```

**Training Concepts:**
- **Hard Pixel Mining**: Focus training on the most difficult pixels to classify
- **KNN Training**: Store features and labels instead of gradient-based training
- **Memory Management**: Clear GPU cache to handle large models

### Prediction Methods

```python
# Lines 175-189: Single-scale prediction
def predict(self, rgb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if self.test_multi_scales is None:
        pred = self(rgb)  # Forward pass (B, num_classes, H, W)
        if not isinstance(self.head, KNeighborsClassifier):
            pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
    else:
        pred = self.multi_scale_test_augmentation(rgb)  # Multi-scale prediction
    
    # Resize to final test output size
    pred = nn.functional.interpolate(pred, size=self.test_output_size, mode='bilinear',
                                     align_corners=False)
    pred = pred.argmax(dim=1)  # Get class predictions (B, H, W)
    
    # Apply mask (e.g., ego car mask)
    if mask is not None:
        pred[mask] = self.ignore_index
    return pred
```

### Multi-Scale Test Augmentation

```python
# Lines 191-229: Multi-scale test augmentation
def multi_scale_test_augmentation(self, rgb: torch.Tensor) -> torch.Tensor:
    all_preds = []  # Store predictions at all scales
    batch_size = rgb.shape[0]
    img_h, img_w = rgb.shape[2:]
    
    for scale in self.test_multi_scales:
        # Calculate split dimensions
        image_h_split, image_w_split = img_h // scale, img_w // scale
        train_output_h_split, train_output_w_split = \
            self.train_output_size[0] // scale, self.train_output_size[1] // scale
        
        # Split image into patches
        rgb_split = torch.split(rgb, image_h_split, dim=2)
        rgb_split = [torch.split(split, image_w_split, dim=3) for split in rgb_split]
        
        # Initialize prediction tensor for this scale
        pred_scale = torch.zeros((batch_size, self.num_classes, 
                                self.train_output_size[0], self.train_output_size[1]))
        
        # Process each split
        for row, row_splits in enumerate(rgb_split):
            for col, rgb_split_i in enumerate(row_splits):
                # Upscale split to full image size
                rgb_split_i_upscaled = T.functional.resize(
                    rgb_split_i, [img_h, img_w], interpolation=InterpolationMode.BILINEAR)
                
                # Get prediction for this split
                pred = self(rgb_split_i_upscaled)  # (B, num_classes, H, W)
                
                # Resize prediction to split output size
                pred = T.functional.resize(
                    pred, [train_output_h_split, train_output_w_split],
                    interpolation=InterpolationMode.BILINEAR)
                
                # Place prediction in appropriate location
                pred_scale[:, :, 
                    row * train_output_h_split:(row + 1) * train_output_h_split,
                    col * train_output_w_split:(col + 1) * train_output_w_split] = pred
        
        all_preds.append(pred_scale)
    
    # Fuse predictions from all scales
    pred = torch.stack(all_preds, dim=1)  # (B, S, num_classes, H, W)
    if not isinstance(self.head, KNeighborsClassifier):
        pred = torch.softmax(pred, dim=2)  # Apply softmax per scale
    pred = pred.mean(dim=1)  # Average across scales (B, num_classes, H, W)
    return pred
```

**Multi-Scale Testing Benefits:**
- **Scale Robustness**: Handles objects at different scales
- **Fine Detail**: Small scales capture fine details
- **Global Context**: Large scales capture global context
- **Ensemble Effect**: Averaging reduces noise and improves robustness

### Visualization and Testing

```python
# Lines 235-254: Visualization method
def plot(self, rgb: np.array, pred: np.array):
    plt.figure(figsize=(20, 6))
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(10, 10)
    
    rgb = rgb.transpose((1, 2, 0))  # Convert from CHW to HWC
    dataset = self.get_dataset()
    pred_color = dataset.class_id_to_color()[pred, :]  # Map predictions to colors
    
    # Show original image
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.grid(False)
    plt.imshow(rgb)
    
    # Show prediction overlay
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.grid(False)
    plt.imshow(rgb)
    plt.imshow(pred_color, cmap='jet', alpha=0.5, interpolation='nearest')
    plt.show()
```

```python
# Lines 256-279: Test step implementation
def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
    rgb = batch['rgb']  # Preprocessed input (B, 3, H, W)
    ego_car_mask = batch.get('ego_car_mask', None)  # Optional ego vehicle mask
    
    pred = self.predict(rgb, ego_car_mask)  # Get semantic predictions (B, H, W)
    
    pred = pred.cpu().numpy()
    rgb_original = batch['rgb_original'].cpu().numpy()  # Original unprocessed image
    
    # Optional visualization
    if self.test_plot:
        for rgb_i, pred_i in zip(rgb_original, pred):
            self.plot(rgb_i, pred_i)
    
    # Optional saving of predictions
    if self.test_save_dir is not None:
        semantic_paths = batch['semantic_path']
        dataset = self.get_dataset()
        dataset_path_base = str(dataset.path_base)
        
        for pred_i, semantic_path in zip(pred, semantic_paths):
            pred_path = semantic_path.replace(dataset_path_base, self.test_save_dir)
            if not os.path.exists(os.path.dirname(pred_path)):
                os.makedirs(os.path.dirname(pred_path))
            pred_img = Image.fromarray(pred_i.astype(np.uint8))
            pred_img.save(pred_path)
```

---

## 🔲 3. Boundary Fine-Tuner (`boundary_fine_tuning.py`)

### Class Definition and Parameters

```python
# Lines 25-62: BoundaryFineTuner class definition and parameters
class BoundaryFineTuner(FineTuner):
    """Fine-tunes a small head on top of the DINOv2 model for boundary estimation."""
```

**Key Parameters:**
- **mode**: 'affinity' (pairwise pixel similarity) or 'direct' (direct boundary prediction)
- **neighbor_radius**: Distance for considering pixels as neighbors (typically 1.5)
- **threshold_boundary**: Threshold for converting probabilities to binary boundaries
- **num_boundary_neighbors**: Minimum neighbors with different labels to be boundary

```python
# Lines 54-125: Head architecture based on mode
def __init__(self, dinov2_vit_model: str, mode: str = 'direct',
             upsample_factor: Optional[float] = None, head: str = 'linear',
             neighbor_radius: float = 1.5, threshold_boundary: float = 0.95,
             num_boundary_neighbors: int = 1,
             test_output_size: Optional[Tuple[int, int]] = None,
             test_multi_scales: Optional[List[int]] = None,
             test_plot: bool = False, use_adapter: bool = False):
    
    super().__init__(dinov2_vit_model=dinov2_vit_model, blocks=None,
                     upsample_factor=upsample_factor, use_adapter=use_adapter)
    
    assert mode in ['affinity', 'direct']
    self.mode = mode
    self.neighbor_radius = neighbor_radius
    self.threshold_boundary = threshold_boundary
    self.num_boundary_neighbors = num_boundary_neighbors
    self.test_output_size = test_output_size
    self.test_multi_scales = test_multi_scales
    self.test_plot = test_plot
```

### Boundary Detection Modes

**Affinity Mode** (Lines 78-90):
```python
if self.mode == 'affinity':
    if self.head == 'mlp':
        # Pairwise feature comparison for boundary detection
        self.head = nn.Sequential(
            nn.Linear(2 * self.feat_dim, 600),  # Input: concatenated feature pairs
            nn.ReLU(),
            nn.Linear(600, 600),
            nn.ReLU(),
            nn.Linear(600, 400),
            nn.ReLU(),
            nn.Linear(400, 1),  # Output: similarity score
        )
```

**Direct Mode** (Lines 91-124):
```python
elif self.mode == 'direct':
    if self.use_adapter:
        feat_dim = 4 * self.feat_dim  # 4x features from adapter
    else:
        feat_dim = self.feat_dim
    
    if head == 'linear':
        # Direct boundary prediction from features
        self.head = nn.Conv2d(feat_dim, 1, kernel_size=1, stride=1, padding=0)
    elif head == 'knn':
        # Non-parametric boundary detection
        self.head = KNeighborsClassifier(n_neighbors=5, leaf_size=10)
        self.knn_X = []
        self.knn_y = []
    # ... similar CNN and MLP architectures
```

### Connected Indices Computation

```python
# Lines 128-165: Connected indices computation for affinity mode
def connected_indices(self, h: int, w: int, batch_size: Optional[int] = None) -> np.ndarray:
    """Returns all connected flattened pixel indices for affinity computation."""
    
    # Check cache to avoid recomputation
    if self.connected_indices_cache is not None and \
            self.connected_indices_cache[0] == h and self.connected_indices_cache[1] == w and \
            self.connected_indices_cache[2] == batch_size:
        return self.connected_indices_cache[3]
    
    # Generate pixel coordinates
    coordinates = np.indices((h, w)).reshape(2, -1).T  # (H*W, 2)
    
    # Find neighbors within radius using sklearn
    graph = radius_neighbors_graph(coordinates, radius=self.neighbor_radius,
                                   mode='connectivity', include_self=False)  # (H*W, H*W)
    
    # Extract connected pixel pairs
    connected_indices = np.argwhere(graph == 1)  # (I, 2) where I is number of connections
    
    # Tile for batch processing if needed
    if batch_size is not None:
        connected_indices = np.tile(connected_indices, (batch_size, 1, 1))  # (B, I, 2)
    
    # Cache result for efficiency
    self.connected_indices_cache = (h, w, batch_size, connected_indices)
    return connected_indices
```

**Connected Indices Concept:**
- Creates pairs of spatially neighboring pixels
- Used in affinity mode to compare features between neighboring pixels
- Cached to avoid recomputation for same image sizes

### Forward Pass

```python
# Lines 167-208: Forward method with mode-specific processing
def forward(self, img: torch.Tensor, connected_indices: np.array = None,
            segment_mask=None) -> torch.Tensor:
    x = self.forward_encoder(img)  # Extract DINOv2 features (B, feat_dim, H, W)
    
    if self.mode == 'affinity':
        # Affinity mode: Compare features between neighboring pixels
        batch_size = x.shape[0]
        h, w = x.shape[2], x.shape[3]
        
        x = x.view((batch_size, self.feat_dim, h * w))  # Flatten spatial dims
        x = x.permute((0, 2, 1)).contiguous()  # (B, H*W, feat_dim)
        
        # Handle optional segment masking
        if segment_mask is not None:
            segment_mask = segment_mask.view(-1)  # (B*H*W)
            x = x.view(-1, self.feat_dim)  # (B*H*W, feat_dim)
            x = x[segment_mask == 1, :]  # Keep only valid pixels
            x = x.view(batch_size, -1, self.feat_dim)  # (B, K, feat_dim)
        
        # Get connected pixel pairs
        if connected_indices is None:
            connected_indices = self.connected_indices(h, w, batch_size)
        
        # Extract features for pixel pairs
        x = x.view(-1, self.feat_dim)  # (B*K, feat_dim)
        connected_indices = connected_indices.reshape(-1, 2)  # (B*I, 2)
        x1 = x[connected_indices[:, 0], :]  # Features of first pixels (B*I, feat_dim)
        x2 = x[connected_indices[:, 1], :]  # Features of second pixels (B*I, feat_dim)
        
        # Concatenate pair features and predict affinity
        x = torch.cat((x1, x2), dim=1)  # (B*I, 2*feat_dim)
        x = x.view(batch_size, -1, 2 * self.feat_dim)  # (B, I, 2*feat_dim)
        x = self.head(x)  # (B, I, 1)
        x = torch.sigmoid(x)  # Convert to probabilities
        
    elif self.mode == 'direct':
        # Direct mode: Predict boundaries directly from features
        if isinstance(self.head, KNeighborsClassifier):
            if self.training:
                return x  # Return features during training
            # KNN inference
            feat_shape = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, feat_shape[1])
            x = x.detach().cpu().numpy()
            x = self.head.predict_proba(x)  # (B*H*W, 2)
            x = np.expand_dims(x[:, 1], axis=1)  # Take positive class probability
            x = torch.from_numpy(x).to(self.device)
            x = x.reshape(feat_shape[0], feat_shape[2], feat_shape[3], -1).permute(0, 3, 1, 2)
        else:
            x = self.head(x)  # Apply parametric head (B, 1, H, W)
            x = torch.sigmoid(x)  # Convert to probabilities
    
    return x
```

**Forward Pass Concepts:**
- **Affinity Mode**: Compares features between neighboring pixels to detect boundaries
- **Direct Mode**: Directly predicts boundary probability for each pixel
- **Sigmoid Activation**: Converts logits to probabilities [0, 1]

### Training Step

```python
# Lines 210-268: Training step with boundary ground truth generation
def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
    rgb = train_batch['rgb']  # Input images
    ins = train_batch['instance'].long()  # Instance labels
    
    device = rgb.device
    batch_size = rgb.shape[0]
    rgb_h, rgb_w = rgb.shape[2:]
    patches_h, patches_w = rgb_h // self.patch_size, rgb_w // self.patch_size
    upsample_factor = 1.0 if self.upsample_factor is None else self.upsample_factor
    network_output_size = (int(patches_h * upsample_factor), int(patches_w * upsample_factor))
    
    # Resize instance labels to network output size
    ins = TF.resize(ins, size=[network_output_size[0], network_output_size[1]],
                    interpolation=TF.InterpolationMode.NEAREST)
    ins_flattened = ins.view(ins.shape[0], -1)  # (B, H*W)
    connected_indices = self.connected_indices(network_output_size[0], network_output_size[1])
    
    if self.mode == 'affinity':
        # Affinity training: same instance = similar (1), different instance = dissimilar (0)
        ins_boundary = (ins_flattened[:, connected_indices[:, 0]] ==
                       ins_flattened[:, connected_indices[:, 1]]).to(torch.float)  # (B, I)
        
        pred = self(rgb)  # (B, I, 1)
        pred = pred.squeeze(2)  # (B, I)
        
    elif self.mode == 'direct':
        # Direct training: Generate boundary ground truth
        # Step 1: Find pixels with different instance neighbors
        ins_boundary = (ins_flattened[:, connected_indices[:, 0]] !=
                       ins_flattened[:, connected_indices[:, 1]]).cpu().numpy().astype(int)  # (B, I)
        
        # Step 2: Aggregate to pixel level - count different neighbors per pixel
        connected_indices_batch = np.tile(connected_indices, (batch_size, 1, 1))  # (B, I, 2)
        indices = connected_indices_batch[:, :, 0]  # (B, I)
        ins_boundary = np.add.reduceat(ins_boundary,
                                      np.unique(indices, return_index=True, axis=1)[1],
                                      axis=1)  # (B, H*W)
        
        # Step 3: Apply threshold - pixel is boundary if it has enough different neighbors
        ins_boundary = np.logical_not(ins_boundary >= self.num_boundary_neighbors)  # (B, H*W)
        # Note: ins_boundary is 0 for boundary pixels, 1 for non-boundary pixels
        
        ins_boundary = torch.Tensor(ins_boundary.reshape(batch_size, network_output_size[0],
                                                         network_output_size[1])).to(device)
        
        if isinstance(self.head, KNeighborsClassifier):
            # KNN training: store features and labels
            x = self(rgb)  # (B, feat_dim, H, W)
            x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # (B*H*W, feat_dim)
            x = x.detach().cpu().numpy()
            self.knn_X.append(x)
            
            ins_boundary = ins_boundary.reshape(-1).detach().cpu().numpy()
            self.knn_y.append(ins_boundary)
        else:
            pred = self(rgb)  # (B, 1, H, W)
            pred = pred.squeeze(1)  # (B, H, W)
    
    # Compute loss
    if isinstance(self.head, KNeighborsClassifier):
        loss = torch.tensor([0.0], requires_grad=True).to(self.device)
    else:
        loss = F.binary_cross_entropy(pred, ins_boundary)
    
    self.log('train_loss', loss)
    return loss
```

**Boundary Ground Truth Generation:**
1. **Instance Comparison**: Compare instance labels between neighboring pixels
2. **Neighbor Counting**: Count how many different instance neighbors each pixel has
3. **Threshold Application**: Pixels with enough different neighbors are boundaries
4. **Binary Labels**: 1 for non-boundary, 0 for boundary pixels

### Prediction Methods

```python
# Lines 270-314: Prediction with multi-scale support
def predict(self, rgb: torch.Tensor) -> torch.Tensor:
    batch_size = rgb.shape[0]
    rgb_h, rgb_w = rgb.shape[2:]
    patches_h, patches_w = rgb_h // self.patch_size, rgb_w // self.patch_size
    upsample_factor = 1.0 if self.upsample_factor is None else self.upsample_factor
    network_output_size = (int(patches_h * upsample_factor), int(patches_w * upsample_factor))
    
    if self.mode == 'affinity':
        # Affinity prediction with aggregation
        if self.test_multi_scales is None:
            pred = self(rgb)  # (B, I, 1)
            pred = pred.squeeze(2).detach().cpu().numpy()  # (B, I)
            
            # Apply threshold
            if self.threshold_boundary is not None:
                pred = (pred > self.threshold_boundary).astype(float)
            
            # Aggregate affinity scores to pixel-level boundary map
            connected_indices = self.connected_indices(network_output_size[0],
                                                      network_output_size[1], batch_size)
            indices = connected_indices[:, :, 0]  # (B, I)
            pred = np.add.reduceat(pred, np.unique(indices, return_index=True, axis=1)[1], axis=1)
            pred = (pred >= self.num_boundary_neighbors).astype(float)
            pred = pred.reshape(batch_size, network_output_size[0], network_output_size[1])
            pred = torch.Tensor(pred).to(rgb.device)
        else:
            raise NotImplementedError
            
    elif self.mode == 'direct':
        # Direct boundary prediction
        if self.test_multi_scales is None:
            pred = self(rgb)  # (B, 1, H, W)
            pred = pred.squeeze(1)  # (B, H, W)
        else:
            pred = self.multi_scale_test_augmentation(rgb, self.test_output_size)
            pred = pred.squeeze(1)  # (B, H, W)
        
        # Apply threshold to get binary boundary map
        pred = (pred > self.threshold_boundary).to(torch.float)  # (B, H, W)
    
    # Resize to final output size
    pred = nn.functional.interpolate(pred.unsqueeze(1), size=self.test_output_size,
                                     mode='nearest').squeeze(1)  # (B, H, W)
    return pred
```

---

## 🎯 4. Instance Clustering (`instance_clustering.py`)

### Panoptic Fusion Overview

The InstanceCluster module combines semantic and boundary predictions to generate panoptic segmentation:

```python
# Lines 24-58: InstanceCluster class definition
class InstanceCluster(pl.LightningModule):
    """Panoptic fusion module that uses the semantic and boundary model to cluster 
    semantic blobs into instances."""
```

**Key Parameters:**
- **structure_connectivity**: Connectivity pattern for connected component analysis
- **instance_min_pixel**: Minimum pixels for valid instances
- **erosion_structure**: Morphological operation structure
- **erosion_iterations**: Number of erosion iterations to separate instances

### Initialization and Model Loading

```python
# Lines 60-105: Model initialization and freezing
def __init__(self, semantic_model: pl.LightningModule, semantic_model_ckpt: str,
             boundary_model: pl.LightningModule, boundary_model_ckpt: str,
             structure_connectivity: List[List[int]], instance_min_pixel: int,
             erosion_structure: List[List[int]], erosion_iterations: int,
             output_size: Tuple[int, int], ignore_index: int = 255,
             test_plot: bool = False, test_save_dir: str = None, test_save_vis: bool = False,
             debug_plot: bool = False):
    
    super().__init__()
    
    # Load semantic model
    self.semantic_model = semantic_model
    semantic_model_ckpt_dict = torch.load(semantic_model_ckpt, map_location='cpu')
    self.semantic_model.load_state_dict(semantic_model_ckpt_dict['state_dict'])
    self.semantic_model.on_load_checkpoint(semantic_model_ckpt_dict)
    
    # Load boundary model
    self.boundary_model = boundary_model
    boundary_model_ckpt_dict = torch.load(boundary_model_ckpt, map_location='cpu')
    self.boundary_model.load_state_dict(boundary_model_ckpt_dict['state_dict'])
    self.boundary_model.on_load_checkpoint(boundary_model_ckpt_dict)
    
    # Share encoder if same DINOv2 model (memory efficiency)
    if self.semantic_model.dinov2_vit_model == self.boundary_model.dinov2_vit_model:
        self.boundary_model.encoder = self.semantic_model.encoder
    
    # Freeze both models
    for param in self.semantic_model.parameters():
        param.requires_grad = False
    for param in self.boundary_model.parameters():
        param.requires_grad = False
    
    # Set up clustering parameters
    self.structure_connectivity = np.array(structure_connectivity)
    self.instance_min_pixel = instance_min_pixel
    if erosion_iterations > 0:
        self.erosion_footprint = [(np.array(erosion_structure), erosion_iterations)]
    else:
        self.erosion_footprint = None
    self.output_size = output_size
    self.ignore_index = ignore_index
```

### Panoptic Prediction Algorithm

```python
# Lines 118-215: Main prediction algorithm
def predict(self, rgb: torch.Tensor, rgb_original: torch.Tensor,
            ego_car_mask: Optional[torch.Tensor] = None) -> Tuple[np.array, np.array, np.array]:
    
    # Step 1: Get semantic predictions
    pred_sem = self.semantic_model.predict(rgb, ego_car_mask)  # (B, H, W)
    pred_sem = pred_sem.cpu().numpy()
    
    # Step 2: Get boundary predictions  
    pred_boundary = self.boundary_model.predict(rgb)  # (B, H, W)
    pred_boundary = pred_boundary.cpu().numpy()
    
    pred_instances_batch = []
    
    # Process each image in batch
    for rgb_i, rgb_original_i, pred_sem_i, pred_boundary_i in zip(rgb, rgb_original, pred_sem, pred_boundary):
        
        if self.debug_plot:
            self.semantic_model.plot(rgb_original_i, pred_sem_i)
        
        assert pred_sem_i.shape == tuple(self.output_size)
        pred_instances = np.zeros(self.output_size, dtype=int)  # Instance ID map
        number_of_instances = 0
        
        # Step 3: Process each "thing" class (classes that have instances)
        thing_classes = self.get_dataset().thing_classes
        for semantic_class_id in thing_classes:
            semantic_class_mask = pred_sem_i == semantic_class_id  # (H, W)
            if np.sum(semantic_class_mask) == 0:  # Skip if class not present
                continue
            
            # Step 4: Connected Component Analysis for semantic segments
            semantic_segments_mask = scipy.ndimage.label(semantic_class_mask,
                                                        structure=self.structure_connectivity)[0]
            
            if self.debug_plot:
                self.plot_instances(rgb_original_i, semantic_segments_mask)
            
            # Process each semantic segment
            for semantic_segment_id in np.unique(semantic_segments_mask):
                if semantic_segment_id == 0:  # Skip background
                    continue
                
                semantic_segment_mask = (semantic_segments_mask == semantic_segment_id)
                
                # Step 5: Filter small segments
                if np.sum(semantic_segment_mask) < self.instance_min_pixel:
                    pred_sem_i[semantic_segment_mask] = self.ignore_index
                    continue
                
                # Step 6: Apply boundary mask to separate instances
                instances_mask = np.logical_and(semantic_segment_mask, pred_boundary_i)
                
                # Step 7: Optional morphological erosion
                if self.erosion_footprint is not None:
                    instances_mask = binary_erosion(instances_mask, footprint=self.erosion_footprint)
                
                # Step 8: Connected Component Analysis for instances
                instances_mask = scipy.ndimage.label(instances_mask, 
                                                   structure=self.structure_connectivity)[0]
                instances_mask = remove_small_objects(instances_mask, 
                                                    min_size=self.instance_min_pixel)
                
                # Step 9: Fallback - if no instances found, use whole segment
                if np.sum(instances_mask) == 0:
                    instances_mask[semantic_segment_mask] = 1
                
                # Step 10: Renumber instances consecutively
                instances_ids = np.unique(instances_mask)
                for i in range(1, len(instances_ids)):
                    instances_mask[instances_mask == instances_ids[i]] = i
                
                # Step 11: Assign remaining semantic pixels to nearest instance (1-NN)
                assert semantic_segment_mask.shape == instances_mask.shape
                coordinates = np.indices((self.output_size[0], self.output_size[1])).reshape(2, -1).T
                coordinates_sem_seg = coordinates[semantic_segment_mask.reshape(-1) == 1]
                coordinates_instances = coordinates[instances_mask.reshape(-1) != 0]
                
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(coordinates_instances, 
                       instances_mask.reshape(-1)[instances_mask.reshape(-1) != 0])
                instances_mask_shape = instances_mask.shape
                instances_mask = instances_mask.reshape(-1)
                instances_mask[semantic_segment_mask.reshape(-1) == 1] = knn.predict(coordinates_sem_seg)
                instances_mask = instances_mask.reshape(instances_mask_shape)
                
                # Step 12: Add to global instance map with unique IDs
                instances_mask += number_of_instances
                instances_mask[instances_mask == number_of_instances] = 0
                pred_instances += instances_mask
                number_of_instances = np.max(pred_instances)
        
        pred_instances_batch.append(pred_instances)
    
    pred_instances = np.stack(pred_instances_batch, axis=0)
    return pred_instances, pred_sem, pred_boundary
```

**Panoptic Algorithm Steps:**
1. **Semantic Prediction**: Get pixel-wise semantic labels
2. **Boundary Prediction**: Get boundary probability map
3. **Thing Class Processing**: Process only classes that have instances
4. **Semantic Clustering**: Use CCA to find connected semantic regions
5. **Size Filtering**: Remove segments smaller than minimum threshold
6. **Boundary Application**: Use boundaries to separate instances within segments
7. **Morphological Operations**: Optional erosion to better separate instances
8. **Instance Clustering**: Use CCA again to find individual instances
9. **Fallback Strategy**: If no instances found, treat whole segment as one instance
10. **Pixel Assignment**: Assign remaining semantic pixels to nearest instance using 1-NN
11. **Global Numbering**: Assign unique instance IDs across the entire image

### Key Concepts and Algorithms

**Connected Component Analysis (CCA):**
- Groups connected pixels with same label
- Uses structure_connectivity to define neighborhood (4-connected vs 8-connected)
- Implemented via `scipy.ndimage.label`

**Morphological Operations:**
- **Erosion**: Shrinks object boundaries to separate touching instances
- **Structure**: Defines shape of morphological kernel
- Helps separate instances that boundaries couldn't fully separate

**K-Nearest Neighbors Assignment:**
- Ensures all semantic pixels are assigned to some instance
- Finds spatially closest instance for unassigned pixels
- Maintains semantic consistency

### Test Step and Saving

```python
# Lines 237-308: Test step with validation and saving
def test_step(self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0):
    rgb = batch['rgb']
    rgb_original = batch['rgb_original'].cpu().numpy()
    ego_car_mask = batch.get('ego_car_mask', None)
    
    # Get panoptic predictions
    pred_instances, pred_sem, pred_boundary = self.predict(rgb, rgb_original, ego_car_mask)
    
    # Validation: Ensure all thing pixels are assigned to instances
    for pred_instances_i, pred_sem_i in zip(pred_instances, pred_sem):
        assert pred_sem_i.shape == pred_instances_i.shape
        thing_classes_mask = np.isin(pred_sem_i, self.get_dataset().thing_classes)
        assert np.all(thing_classes_mask == (pred_instances_i != 0))
    
    # Optional visualization
    if self.test_plot:
        for rgb_original_i, pred_sem_i, pred_instances_i in zip(rgb_original, pred_sem, pred_instances):
            self.plot_instances(rgb_original_i, pred_instances_i)
            self.semantic_model.plot(rgb_original_i, pred_sem_i)
    
    # Save predictions in dataset format
    if self.test_save_dir is not None:
        semantic_path = batch['semantic_path']
        instance_path = batch['instance_path']
        dataset = self.get_dataset()
        dataset_path_base = str(dataset.path_base)
        
        for pred_sem_i, pred_instances_i, pred_boundary_i, semantic_path_i, instance_path_i in \
                zip(pred_sem, pred_instances, pred_boundary, semantic_path, instance_path):
            
            # Convert to dataset's ground truth format
            pred_sem_i_gt_format, pred_panoptic_i_gt_format = \
                dataset.compute_panoptic_label_in_gt_format(pred_sem_i, pred_instances_i)
            
            # Save semantic predictions
            pred_sem_i_path = semantic_path_i.replace(dataset_path_base, self.test_save_dir)
            if not os.path.exists(os.path.dirname(pred_sem_i_path)):
                os.makedirs(os.path.dirname(pred_sem_i_path))
            pred_img = Image.fromarray(pred_sem_i_gt_format.astype(np.uint8))
            pred_img.save(pred_sem_i_path)
            
            # Save panoptic predictions
            pred_panoptic_i_path = instance_path_i.replace(dataset_path_base, self.test_save_dir)
            if not os.path.exists(os.path.dirname(pred_panoptic_i_path)):
                os.makedirs(os.path.dirname(pred_panoptic_i_path))
            pred_img = Image.fromarray(pred_panoptic_i_gt_format.astype(np.uint16))
            pred_img.save(pred_panoptic_i_path)
            
            # Optional visualization saving
            if self.test_save_vis:
                # Generate colored visualizations
                pred_sem_i_color = self.get_dataset().class_id_to_color()[pred_sem_i, :]
                pred_ins_i_color = self.id_color_array[pred_instances_i, :]
                pred_panop_i_color = np.zeros_like(pred_sem_i_color)
                pred_panop_i_color[pred_instances_i == 0, :] = pred_sem_i_color[pred_instances_i == 0, :]
                pred_panop_i_color[pred_instances_i != 0, :] = pred_ins_i_color[pred_instances_i != 0, :]
                
                # Save colored images
                pred_img = Image.fromarray(pred_sem_i_color)
                pred_sem_i_color_path = pred_sem_i_path.replace('.png', '_color.png')
                pred_img.save(pred_sem_i_color_path)
                
                pred_img = Image.fromarray(pred_panop_i_color)
                pred_panop_i_color_path = pred_panoptic_i_path.replace('.png', '_color.png')
                pred_img.save(pred_panop_i_color_path)
                
                if pred_boundary_i is not None:
                    pred_boundary_i_path = pred_panoptic_i_path.replace('.png', '_boundary.png')
                    pred_img = Image.fromarray(pred_boundary_i.astype(np.uint8) * 255).convert('RGB')
                    pred_img.save(pred_boundary_i_path)
```

---

## 🚀 Complete Execution Flow Summary

### 1. Data Input and Preprocessing
```
RGB Image (H, W, 3) → Normalization → Resizing → (B, 3, H, W)
```

### 2. Feature Extraction (DINOv2)
```
Input (B, 3, H, W) 
→ Patch Embedding (B, num_patches, feat_dim)
→ Transformer Blocks 
→ Feature Maps (B, feat_dim, H/14, W/14)
→ Optional Upsampling
```

### 3. Parallel Head Processing
```
                     Features
                    /         \
            Semantic Head    Boundary Head
                 |              |
          Class Logits    Boundary Probs
         (B, C, H, W)     (B, 1, H, W)
```

### 4. Panoptic Fusion
```
Semantic Map + Boundary Map
→ Connected Component Analysis
→ Boundary-based Instance Separation  
→ Morphological Operations
→ K-NN Pixel Assignment
→ Panoptic Segmentation (B, H, W)
```

### 5. Key Training Concepts

**Semantic Training:**
- Cross-entropy loss with ignore index
- Hard pixel mining for difficult examples
- Multi-scale test augmentation

**Boundary Training:**
- Binary cross-entropy loss
- Ground truth from instance label differences
- Neighbor-based boundary detection

**Few-Shot Learning:**
- DINOv2 features frozen (pretrained)
- Only lightweight heads trained
- K-NN as non-parametric alternative

This comprehensive documentation covers every major component and concept in the panoptic label generator, explaining how DINOv2 features are extracted and processed through semantic and boundary heads to create panoptic segmentation labels.