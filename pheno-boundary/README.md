# Pheno-Boundary: Dynamic Agricultural Parcel Delineation and Stability Analysis

Multi-year field boundary detection and temporal stability monitoring using Sentinel-2 imagery and the Fields of The World (FTW) neural network.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Technical Architecture](#technical-architecture)
4. [Data Sources](#data-sources)
5. [Environment Setup](#environment-setup)
6. [Project Structure](#project-structure)
7. [Implementation Phases](#implementation-phases)
8. [Usage Guide](#usage-guide)
9. [Expected Outputs](#expected-outputs)
10. [Team & Task Allocation](#team--task-allocation)
11. [References](#references)

---

## Project Overview

Agricultural parcels are dynamic functional units that merge, split, and change due to management practices such as strip cropping, seasonal rotation, and land consolidation. Traditional approaches treat field boundaries as static features derived from single-date imagery or cadastral records.

**Pheno-Boundary** addresses this limitation by:

- Generating annual field boundary masks using the FTW pre-trained neural network
- Computing temporal stability metrics (IoU) across multiple years
- Detecting and visualizing boundary change events
- Quantifying cultivated land stability for regenerative agriculture verification

### Study Area

- **Location**: South Tyrol, Italy (Alpine agricultural valley)
- **Bounding Box**: `[11.290770, 46.356466, 11.315060, 46.389037]`
- **Extent**: ~2.4 km × 3.6 km
- **Terrain**: Alpine valley with orchards, vineyards, hay meadows

### Temporal Coverage

- **Period**: 2020-2024 (4+ years)
- **Imagery**: Sentinel-2 L2A (atmospherically corrected)
- **Cadence**: All available cloud-free acquisitions

---

## Objectives

### Primary

1. Implement multi-year field boundary segmentation using FTW baseline model
2. Develop IoU-based stability metrics for boundary change detection
3. Create reproducible notebook demonstrating the complete workflow

### Secondary

1. Validate FTW model performance on alpine terrain (outside original training distribution)
2. Generate temporal variance metrics (NDVI_STD, NDVI_CV) as supplementary features
3. Produce publication-ready visualizations of stability analysis

### Success Criteria

| Metric | Target |
|--------|--------|
| Years processed | ≥ 4 |
| Inference completion | All years successfully segmented |
| IoU matrix computed | Pairwise comparison for all years |
| Change map generated | Binary mask of unstable boundaries |

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                              │
│  EOPF STAC Catalog → Sentinel-2 L2A Zarr → Multi-temporal Datacube  │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PREPROCESSING                                  │
│  Cloud Masking (SCL) → Band Selection → Coordinate Transformation   │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SEASONAL COMPOSITING                              │
│  Spring Median (Mar-May) + Summer Median (Jul-Sep) → 8-channel input│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FTW MODEL INFERENCE                              │
│  U-Net + EfficientNet-b3 → Field Mask + Boundary Probability        │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STABILITY ANALYSIS                                │
│  Pairwise IoU → Stability Matrix → Change Detection → Visualization │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Selection Rationale

**Fields of The World (FTW) Baseline** selected over alternatives:

| Criterion | FTW | VITO U-Net | Delineate Anything |
|-----------|-----|------------|-------------------|
| Austria in training data | ✓ | ✗ | ✓ |
| Native Sentinel-2 support | ✓ | ✓ | ✗ (RGB only) |
| Bi-temporal input design | ✓ | ✗ | ✗ |
| Semantic segmentation output | ✓ | ✓ | ✗ (instance) |
| IoU computation simplicity | Easy | Easy | Complex |

### FTW Model Specifications

- **Architecture**: U-Net with EfficientNet-b3 encoder
- **Input**: 8 channels (B02, B03, B04, B08 × 2 timestamps)
- **Output**: 3-class segmentation (field interior, boundary, non-field)
- **Resolution**: 10m (native Sentinel-2)
- **Weights**: `prue_efnetb3_ccby_checkpoint.ckpt` (v3.1)

---

## Data Sources

### Sentinel-2 L2A via EOPF

- **Catalog**: `https://stac.core.eopf.eodc.eu`
- **Collection**: `sentinel-2-l2a`
- **Format**: Zarr (cloud-optimized)
- **Bands Used**:

| Band | Wavelength | Resolution | Purpose |
|------|------------|------------|---------|
| B02 | 490 nm (Blue) | 10m | RGB visualization, FTW input |
| B03 | 560 nm (Green) | 10m | RGB visualization, FTW input |
| B04 | 665 nm (Red) | 10m | NDVI calculation, FTW input |
| B08 | 842 nm (NIR) | 10m | NDVI calculation, FTW input |
| SCL | N/A | 20m | Cloud/shadow masking |

### FTW Pre-trained Weights

- **Source**: `github.com/fieldsoftheworld/ftw-baselines/releases/tag/v3.1`
- **Model File**: `prue_efnetb3_ccby_checkpoint.ckpt`
- **Alternative (v1)**: `3_Class_FULL_FTW_Pretrained.ckpt`
- **Hugging Face**: `huggingface.co/torchgeo/fields-of-the-world`
- **License**: Apache 2.0

---

## Environment Setup

### Option A: Google Colab (Recommended for Inference)

```python
# Cell 1: Install dependencies
!pip install -q \
    xarray \
    rioxarray \
    zarr \
    dask \
    distributed \
    pystac-client \
    pyproj \
    geopandas \
    matplotlib \
    seaborn \
    torch \
    torchvision \
    segmentation-models-pytorch \
    ftw-tools

# Cell 2: Verify GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Option B: Local Development (32GB RAM)

```bash
# Create conda environment
conda create -n pheno-boundary python=3.11 -y
conda activate pheno-boundary

# Install core dependencies
pip install xarray rioxarray zarr dask distributed
pip install pystac-client pyproj geopandas
pip install matplotlib seaborn folium
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install segmentation-models-pytorch
pip install ftw-tools

# Development tools
pip install jupyterlab ipywidgets tqdm
```

### Option C: Kaggle Notebook

```python
# Enable GPU in Settings → Accelerator → GPU T4 x2
# Dependencies pre-installed, add missing:
!pip install -q pystac-client rioxarray ftw-tools
```

### Hardware Requirements

| Environment | RAM | GPU | Storage | Use Case |
|-------------|-----|-----|---------|----------|
| Local | 32GB | None | 50GB | Development, debugging |
| Colab T4 | 12.7GB | 15GB VRAM | 78GB | Inference |
| Kaggle T4 | 13GB | 15GB VRAM | 20GB | Inference backup |

---

## Project Structure

```
pheno-boundary/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # STAC query, datacube inspection
│   ├── 02_preprocessing.ipynb         # Cloud masking, compositing
│   ├── 03_ftw_inference.ipynb         # Model inference (run on Colab)
│   ├── 04_stability_analysis.ipynb    # IoU computation, visualization
│   └── 05_full_pipeline.ipynb         # Complete integrated workflow
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # EOPF data access functions
│   ├── preprocessing.py         # Cloud masking, compositing logic
│   ├── inference.py             # FTW model wrapper
│   ├── stability.py             # IoU and change detection metrics
│   └── visualization.py         # Plotting utilities
│
├── configs/
│   └── config.yaml              # AOI, temporal range, model paths
│
├── data/
│   ├── raw/                     # Downloaded Zarr data (gitignored)
│   ├── processed/               # Seasonal composites (gitignored)
│   └── outputs/                 # Masks, metrics, figures
│
├── models/
│   └── ftw/                     # Downloaded FTW weights (gitignored)
│
└── docs/
    ├── methodology.md           # Detailed technical approach
    └── results.md               # Analysis findings
```

---

## Implementation Phases

### Phase 1: Setup and Data Access (Week 1)

**Owner**: Ousama Bin Zamir

#### Task 1.1: Repository Initialization

```bash
# Clone and setup
git clone https://github.com/<username>/pheno-boundary.git
cd pheno-boundary
conda activate pheno-boundary
```

#### Task 1.2: STAC Connection and Data Discovery

```python
# notebooks/01_data_exploration.ipynb

import pystac_client

# Connect to EOPF catalog
catalog = pystac_client.Client.open("https://stac.core.eopf.eodc.eu")

# Define AOI
bbox = [11.290770, 46.356466, 11.315060, 46.389037]

# Query available data
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime=["2019-12-31", "2024-12-31"],
)
items = list(search.items_as_dicts())
print(f"Total scenes found: {len(items)}")

# Inspect temporal distribution
import pandas as pd
dates = [item['properties']['datetime'] for item in items]
df = pd.DataFrame({'date': pd.to_datetime(dates)})
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
print(df.groupby('year').size())
```

#### Task 1.3: Datacube Formation

```python
# src/data_loader.py

import xarray as xr
from pyproj import Transformer
import dask

def reproject_bbox(bbox, src_crs="EPSG:4326", dst_crs="EPSG:32632"):
    """Transform bounding box from WGS84 to UTM."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xmin, ymin = transformer.transform(bbox[0], bbox[1])
    xmax, ymax = transformer.transform(bbox[2], bbox[3])
    return [xmin, ymin, xmax, ymax]

def load_single_scene(item_dict, bbox_ll, bands=['b02', 'b03', 'b04', 'b08']):
    """Load and crop a single Sentinel-2 scene."""
    href = item_dict['assets']['SR_10m']['href']
    base_path = href.split('/measurements')[0]
    
    ds = xr.open_datatree(base_path, engine="zarr", chunks={}, mask_and_scale=True)
    
    dst_crs = item_dict['properties'].get("proj:code", "EPSG:32632")
    utm_bbox = reproject_bbox(bbox_ll, dst_crs=dst_crs)
    
    # Load 10m bands
    ds_ref = (
        ds["measurements"]["reflectance"]["r10m"]
        .to_dataset()[bands]
        .sel(x=slice(utm_bbox[0], utm_bbox[2]), y=slice(utm_bbox[3], utm_bbox[1]))
        .expand_dims(time=[item_dict['properties']['datetime']])
    )
    
    # Load SCL mask (20m resolution)
    ds_scl = (
        ds["conditions"]["mask"]["l2a_classification"]["r20m"]
        .to_dataset()
        .sel(x=slice(utm_bbox[0], utm_bbox[2]), y=slice(utm_bbox[3], utm_bbox[1]))
        .expand_dims(time=[item_dict['properties']['datetime']])
    )
    
    # Interpolate SCL to 10m grid
    ds_scl_interp = ds_scl.interp(x=ds_ref.x, y=ds_ref.y, method="nearest")
    
    return xr.merge([ds_ref, ds_scl_interp])

def build_datacube(items, bbox, parallel=True):
    """Build multi-temporal datacube from STAC items."""
    if parallel:
        delayed_results = [dask.delayed(load_single_scene)(item, bbox) for item in items]
        results = dask.compute(*delayed_results)
    else:
        results = [load_single_scene(item, bbox) for item in items]
    
    crs = items[0]['properties'].get("proj:code", "EPSG:32632")
    datacube = xr.concat(results, dim="time").sortby("time")
    datacube = datacube.rio.write_crs(crs)
    
    return datacube
```

#### Task 1.4: FTW Model Download and Verification

```python
# notebooks/01_data_exploration.ipynb (continued)

import urllib.request
import os

# Create model directory
os.makedirs("models/ftw", exist_ok=True)

# Download FTW weights (v3.1 - prue architecture with EfficientNet-b3)
model_url = "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v3.1/prue_efnetb3_ccby_checkpoint.ckpt"
model_path = "models/ftw/prue_efnetb3_ccby_checkpoint.ckpt"

if not os.path.exists(model_path):
    print("Downloading FTW model weights...")
    urllib.request.urlretrieve(model_url, model_path)
    print(f"Downloaded to {model_path}")

# Verify model loads
import torch
checkpoint = torch.load(model_path, map_location='cpu')
print(f"Model keys: {checkpoint.keys()}")
```

#### Milestone 1 Checklist

- [ ] Repository created and cloned
- [ ] STAC connection verified
- [ ] Datacube loads successfully for all years
- [ ] FTW model weights downloaded
- [ ] Model loads without errors

---

### Phase 2: Preprocessing and Feature Engineering (Week 2)

**Owner**: Ahmad Ahmad Abubakar

#### Task 2.1: Cloud Masking

```python
# src/preprocessing.py

import numpy as np
import xarray as xr

# SCL classification codes
SCL_INVALID = [0, 1, 3, 7, 8, 9, 10]
# 0: NO_DATA, 1: SATURATED, 3: CLOUD_SHADOW
# 7: CLOUD_LOW, 8: CLOUD_MEDIUM, 9: CLOUD_HIGH, 10: THIN_CIRRUS

def apply_cloud_mask(datacube, scl_var='scl'):
    """Mask invalid pixels using Scene Classification Layer."""
    valid_mask = ~datacube[scl_var].isin(SCL_INVALID)
    masked = datacube.where(valid_mask)
    return masked
```

#### Task 2.2: Seasonal Compositing

```python
# src/preprocessing.py (continued)

def assign_season(month):
    """Map month to meteorological season."""
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'

def create_seasonal_composite(datacube, year, season, method='median'):
    """
    Create cloud-free composite for a specific year and season.
    
    Parameters
    ----------
    datacube : xr.Dataset
        Multi-temporal datacube with cloud mask applied
    year : int
        Target year
    season : str
        One of 'spring', 'summer', 'autumn', 'winter'
    method : str
        Aggregation method ('median', 'mean', 'max')
    
    Returns
    -------
    xr.Dataset
        Seasonal composite
    """
    import pandas as pd
    
    season_months = {
        'spring': [3, 4, 5],
        'summer': [6, 7, 8],
        'autumn': [9, 10, 11],
        'winter': [12, 1, 2]
    }
    
    months = season_months[season]
    
    # Handle winter crossing year boundary
    if season == 'winter':
        time_mask = (
            ((datacube.time.dt.year == year) & (datacube.time.dt.month == 12)) |
            ((datacube.time.dt.year == year + 1) & (datacube.time.dt.month.isin([1, 2])))
        )
    else:
        time_mask = (
            (datacube.time.dt.year == year) & 
            (datacube.time.dt.month.isin(months))
        )
    
    subset = datacube.where(time_mask, drop=True)
    
    if len(subset.time) == 0:
        raise ValueError(f"No data found for {year} {season}")
    
    if method == 'median':
        composite = subset.median(dim='time', skipna=True)
    elif method == 'mean':
        composite = subset.mean(dim='time', skipna=True)
    elif method == 'max':
        composite = subset.max(dim='time', skipna=True)
    
    composite.attrs['year'] = year
    composite.attrs['season'] = season
    composite.attrs['n_images'] = len(subset.time)
    
    return composite
```

#### Task 2.3: FTW Input Preparation

```python
# src/preprocessing.py (continued)

def prepare_ftw_input(datacube, year):
    """
    Prepare 8-channel input tensor for FTW model.
    
    FTW expects:
    - Channels 0-3: B02, B03, B04, B08 from planting window (spring)
    - Channels 4-7: B02, B03, B04, B08 from harvest window (summer)
    
    Parameters
    ----------
    datacube : xr.Dataset
        Cloud-masked multi-temporal datacube
    year : int
        Target year
    
    Returns
    -------
    np.ndarray
        Shape (8, H, W) normalized to [0, 1]
    """
    # Create spring composite (planting window)
    spring = create_seasonal_composite(datacube, year, 'spring', method='median')
    
    # Create summer composite (harvest window)
    summer = create_seasonal_composite(datacube, year, 'summer', method='median')
    
    # Extract bands in FTW order
    bands = ['b02', 'b03', 'b04', 'b08']
    
    spring_stack = np.stack([spring[b].values for b in bands], axis=0)
    summer_stack = np.stack([summer[b].values for b in bands], axis=0)
    
    # Concatenate: [B02_s, B03_s, B04_s, B08_s, B02_h, B03_h, B04_h, B08_h]
    ftw_input = np.concatenate([spring_stack, summer_stack], axis=0)
    
    # Normalize to [0, 1] - Sentinel-2 L2A reflectance typically 0-0.5
    ftw_input = np.clip(ftw_input / 0.5, 0, 1).astype(np.float32)
    
    # Handle NaN values
    ftw_input = np.nan_to_num(ftw_input, nan=0.0)
    
    return ftw_input
```

#### Task 2.4: NDVI Temporal Statistics

```python
# src/preprocessing.py (continued)

def compute_ndvi(datacube, red='b04', nir='b08'):
    """Compute NDVI from datacube."""
    ndvi = (datacube[nir] - datacube[red]) / (datacube[nir] + datacube[red])
    return ndvi.clip(-1, 1)

def compute_temporal_statistics(datacube, year):
    """
    Compute NDVI temporal statistics for edge detection.
    
    High temporal variance indicates field boundaries due to:
    - Mixed pixel effects at edges
    - Different phenology between adjacent fields
    
    Returns
    -------
    xr.Dataset
        Contains ndvi_mean, ndvi_std, ndvi_cv
    """
    # Filter to growing season of target year
    growing_season = datacube.where(
        (datacube.time.dt.year == year) & 
        (datacube.time.dt.month.isin([4, 5, 6, 7, 8, 9])),
        drop=True
    )
    
    ndvi = compute_ndvi(growing_season)
    
    stats = xr.Dataset({
        'ndvi_mean': ndvi.mean(dim='time', skipna=True),
        'ndvi_std': ndvi.std(dim='time', skipna=True),
        'ndvi_cv': ndvi.std(dim='time', skipna=True) / ndvi.mean(dim='time', skipna=True).abs(),
        'ndvi_range': ndvi.max(dim='time') - ndvi.min(dim='time'),
    })
    
    return stats
```

#### Milestone 2 Checklist

- [ ] Cloud masking verified visually
- [ ] Seasonal composites generated for all years
- [ ] FTW input tensors validated (shape, range)
- [ ] NDVI statistics computed and saved

---

### Phase 3: Model Inference (Week 2-3)

**Owner**: Ousama Bin Zamir

**Note**: Run this phase on Google Colab with T4 GPU.

#### Task 3.1: FTW Model Wrapper

```python
# src/inference.py

import torch
import numpy as np
import segmentation_models_pytorch as smp

class FTWInference:
    """Wrapper for Fields of The World segmentation model."""
    
    def __init__(self, checkpoint_path, device=None):
        """
        Initialize FTW model.
        
        Parameters
        ----------
        checkpoint_path : str
            Path to FTW_v1_3_Class_FULL.ckpt
        device : str, optional
            'cuda' or 'cpu'. Auto-detect if None.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Build model architecture
        self.model = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            in_channels=8,  # Bi-temporal Sentinel-2
            classes=3,      # Field, boundary, non-field
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"FTW model loaded on {self.device}")
    
    def predict(self, input_array, tile_size=256, overlap=32):
        """
        Run inference with tiled prediction for large images.
        
        Parameters
        ----------
        input_array : np.ndarray
            Shape (8, H, W), values in [0, 1]
        tile_size : int
            Size of inference tiles
        overlap : int
            Overlap between adjacent tiles
        
        Returns
        -------
        dict
            'field_mask': Binary field interior mask
            'boundary_mask': Field boundary probability
            'class_map': 3-class segmentation
        """
        _, H, W = input_array.shape
        
        # Pad to multiple of tile_size
        pad_h = (tile_size - H % tile_size) % tile_size
        pad_w = (tile_size - W % tile_size) % tile_size
        
        padded = np.pad(
            input_array,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode='reflect'
        )
        
        _, pH, pW = padded.shape
        
        # Output arrays
        output_sum = np.zeros((3, pH, pW), dtype=np.float32)
        count = np.zeros((pH, pW), dtype=np.float32)
        
        stride = tile_size - overlap
        
        with torch.no_grad():
            for y in range(0, pH - tile_size + 1, stride):
                for x in range(0, pW - tile_size + 1, stride):
                    tile = padded[:, y:y+tile_size, x:x+tile_size]
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).to(self.device)
                    
                    logits = self.model(tile_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    output_sum[:, y:y+tile_size, x:x+tile_size] += probs
                    count[y:y+tile_size, x:x+tile_size] += 1
        
        # Average overlapping predictions
        output = output_sum / np.maximum(count, 1)
        
        # Crop to original size
        output = output[:, :H, :W]
        
        # Extract outputs
        # Class 0: Non-field, Class 1: Field interior, Class 2: Boundary
        results = {
            'field_prob': output[1],
            'boundary_prob': output[2],
            'class_map': np.argmax(output, axis=0),
            'field_mask': output[1] > 0.5,
            'raw_probs': output,
        }
        
        return results
```

#### Task 3.2: Batch Inference Pipeline

```python
# src/inference.py (continued)

def run_multi_year_inference(datacube, model, years, output_dir):
    """
    Run FTW inference for multiple years.
    
    Parameters
    ----------
    datacube : xr.Dataset
        Cloud-masked multi-temporal datacube
    model : FTWInference
        Loaded model
    years : list
        Years to process
    output_dir : str
        Directory to save results
    
    Returns
    -------
    dict
        {year: results_dict} for each year
    """
    import os
    from src.preprocessing import prepare_ftw_input
    
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    
    for year in years:
        print(f"\nProcessing {year}...")
        
        try:
            # Prepare input
            input_array = prepare_ftw_input(datacube, year)
            print(f"  Input shape: {input_array.shape}")
            
            # Run inference
            results = model.predict(input_array)
            print(f"  Inference complete. Field pixels: {results['field_mask'].sum()}")
            
            # Save results
            np.savez_compressed(
                os.path.join(output_dir, f"ftw_results_{year}.npz"),
                field_prob=results['field_prob'],
                boundary_prob=results['boundary_prob'],
                class_map=results['class_map'],
                field_mask=results['field_mask'],
            )
            
            all_results[year] = results
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[year] = None
    
    return all_results
```

#### Task 3.3: Colab Inference Notebook

```python
# notebooks/03_ftw_inference.ipynb

# Cell 1: Setup
!pip install -q segmentation-models-pytorch

import torch
print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")

# Cell 2: Mount Drive (for data persistence)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Download model (v3.1)
!mkdir -p models/ftw
!wget -q -O models/ftw/prue_efnetb3_ccby_checkpoint.ckpt \
    https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v3.1/prue_efnetb3_ccby_checkpoint.ckpt

# Cell 4: Load model
from src.inference import FTWInference
model = FTWInference("models/ftw/prue_efnetb3_ccby_checkpoint.ckpt", device='cuda')

# Cell 5: Load preprocessed data (from Drive)
import numpy as np
datacube = ...  # Load from saved Zarr or NetCDF

# Cell 6: Run inference
years = [2020, 2021, 2022, 2023]
results = run_multi_year_inference(datacube, model, years, "data/outputs/masks")

# Cell 7: Save to Drive for local analysis
!cp -r data/outputs/masks /content/drive/MyDrive/pheno-boundary/
```

#### Milestone 3 Checklist

- [ ] FTW model loads on GPU
- [ ] Single year inference succeeds
- [ ] All years processed
- [ ] Results saved to persistent storage

---

### Phase 4: Stability Analysis (Week 3-4)

**Owner**: Ahmad Ahmad Abubakar

#### Task 4.1: IoU Computation

```python
# src/stability.py

import numpy as np
import pandas as pd

def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union between two binary masks.
    
    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Binary masks (boolean or 0/1)
    
    Returns
    -------
    float
        IoU score in [0, 1]
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def compute_stability_matrix(masks_dict):
    """
    Compute pairwise IoU matrix for all years.
    
    Parameters
    ----------
    masks_dict : dict
        {year: binary_mask} for each year
    
    Returns
    -------
    pd.DataFrame
        Symmetric IoU matrix
    """
    years = sorted(masks_dict.keys())
    n = len(years)
    iou_matrix = np.zeros((n, n))
    
    for i, y1 in enumerate(years):
        for j, y2 in enumerate(years):
            iou_matrix[i, j] = compute_iou(masks_dict[y1], masks_dict[y2])
    
    return pd.DataFrame(iou_matrix, index=years, columns=years)

def compute_boundary_iou(boundaries_dict):
    """IoU specifically for boundary predictions (narrower structures)."""
    return compute_stability_matrix(boundaries_dict)
```

#### Task 4.2: Change Detection

```python
# src/stability.py (continued)

def detect_boundary_changes(mask_prev, mask_curr, threshold=0.5):
    """
    Identify pixels where field boundaries changed.
    
    Returns
    -------
    dict
        'added': New field areas (was non-field, now field)
        'removed': Lost field areas (was field, now non-field)
        'stable': Unchanged areas
        'changed': Any change
    """
    prev = mask_prev > threshold
    curr = mask_curr > threshold
    
    return {
        'added': np.logical_and(~prev, curr),      # Field expansion
        'removed': np.logical_and(prev, ~curr),    # Field contraction
        'stable': prev == curr,                    # No change
        'changed': prev != curr,                   # Any change
    }

def compute_change_statistics(changes_dict):
    """Summarize change detection results."""
    total_pixels = changes_dict['stable'].size
    
    stats = {
        'total_pixels': total_pixels,
        'added_pixels': changes_dict['added'].sum(),
        'removed_pixels': changes_dict['removed'].sum(),
        'stable_pixels': changes_dict['stable'].sum(),
        'changed_pixels': changes_dict['changed'].sum(),
        'added_pct': 100 * changes_dict['added'].sum() / total_pixels,
        'removed_pct': 100 * changes_dict['removed'].sum() / total_pixels,
        'stability_pct': 100 * changes_dict['stable'].sum() / total_pixels,
    }
    
    return stats

def multi_year_change_analysis(masks_dict):
    """
    Analyze changes between consecutive years.
    
    Returns
    -------
    pd.DataFrame
        Change statistics for each year transition
    """
    years = sorted(masks_dict.keys())
    transitions = []
    
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        changes = detect_boundary_changes(masks_dict[y1], masks_dict[y2])
        stats = compute_change_statistics(changes)
        stats['from_year'] = y1
        stats['to_year'] = y2
        transitions.append(stats)
    
    return pd.DataFrame(transitions)
```

#### Task 4.3: Stability Classification

```python
# src/stability.py (continued)

def classify_stability_zones(masks_dict, stable_threshold=0.8):
    """
    Classify each pixel by temporal stability.
    
    Parameters
    ----------
    masks_dict : dict
        {year: field_mask} for each year
    stable_threshold : float
        Fraction of years a pixel must be consistent
    
    Returns
    -------
    np.ndarray
        Classification map:
        0 = Never field
        1 = Always field (stable)
        2 = Sometimes field (unstable)
    """
    years = sorted(masks_dict.keys())
    n_years = len(years)
    
    # Stack all masks
    stack = np.stack([masks_dict[y].astype(float) for y in years], axis=0)
    
    # Count years as field
    field_frequency = stack.mean(axis=0)
    
    # Classify
    classification = np.zeros_like(field_frequency, dtype=np.uint8)
    classification[field_frequency >= stable_threshold] = 1  # Stable field
    classification[(field_frequency > 0) & (field_frequency < stable_threshold)] = 2  # Unstable
    # 0 remains for never-field
    
    return classification, field_frequency
```

#### Milestone 4 Checklist

- [ ] IoU matrix computed
- [ ] Year-to-year changes quantified
- [ ] Stability zones classified
- [ ] Results exported for visualization

---

### Phase 5: Visualization (Week 4)

**Owner**: Both team members

#### Task 5.1: Visualization Functions

```python
# src/visualization.py

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

def plot_stability_matrix(stability_df, output_path=None):
    """Plot IoU stability matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        stability_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={'label': 'IoU Score'}
    )
    
    ax.set_title('Field Boundary Stability (IoU) Across Years')
    ax.set_xlabel('Year')
    ax.set_ylabel('Year')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_multi_year_masks(masks_dict, extent=None, output_path=None):
    """Plot field masks for all years in grid."""
    years = sorted(masks_dict.keys())
    n = len(years)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_2d(axes)
    
    for idx, year in enumerate(years):
        ax = axes[idx // cols, idx % cols]
        
        im = ax.imshow(masks_dict[year], cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f'{year}')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis('off')
    
    fig.suptitle('Field Masks by Year', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_change_map(changes_dict, year1, year2, output_path=None):
    """Visualize boundary changes between two years."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create RGB change map
    change_map = np.zeros((*changes_dict['stable'].shape, 3), dtype=np.uint8)
    
    # Green = stable field
    change_map[changes_dict['stable'] & (changes_dict['added'] == False)] = [34, 139, 34]
    
    # Blue = added (field expansion)
    change_map[changes_dict['added']] = [30, 144, 255]
    
    # Red = removed (field contraction)
    change_map[changes_dict['removed']] = [220, 20, 60]
    
    # Gray = stable non-field
    stable_nonfield = changes_dict['stable'] & ~changes_dict['added'] & ~changes_dict['removed']
    # Already black (0,0,0)
    
    ax.imshow(change_map)
    ax.set_title(f'Field Boundary Changes: {year1} → {year2}')
    ax.axis('off')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='forestgreen', label='Stable Field'),
        mpatches.Patch(color='dodgerblue', label='Field Added'),
        mpatches.Patch(color='crimson', label='Field Removed'),
        mpatches.Patch(color='black', label='Non-field'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_stability_zones(classification, frequency, output_path=None):
    """Visualize temporal stability classification."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Stability classification
    ax1 = axes[0]
    cmap = plt.cm.colors.ListedColormap(['#808080', '#228B22', '#FFD700'])
    im1 = ax1.imshow(classification, cmap=cmap, vmin=0, vmax=2)
    ax1.set_title('Stability Classification')
    ax1.axis('off')
    
    legend_elements = [
        mpatches.Patch(color='#808080', label='Never Field'),
        mpatches.Patch(color='#228B22', label='Stable Field'),
        mpatches.Patch(color='#FFD700', label='Unstable/Changing'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # Field frequency
    ax2 = axes[1]
    im2 = ax2.imshow(frequency, cmap='YlGn', vmin=0, vmax=1)
    ax2.set_title('Field Presence Frequency')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, label='Fraction of Years')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
```

---

## Usage Guide

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/<username>/pheno-boundary.git
cd pheno-boundary

# 2. Setup environment
conda env create -f environment.yml
conda activate pheno-boundary

# 3. Run data exploration locally
jupyter lab notebooks/01_data_exploration.ipynb

# 4. Run inference on Colab
# Upload notebooks/03_ftw_inference.ipynb to Colab
# Enable GPU runtime
# Execute all cells

# 5. Run analysis locally
jupyter lab notebooks/04_stability_analysis.ipynb
```

### Configuration

Edit `configs/config.yaml`:

```yaml
# Area of interest
aoi:
  bbox: [11.290770, 46.356466, 11.315060, 46.389037]
  name: "South Tyrol"

# Temporal range
temporal:
  start_date: "2019-12-31"
  end_date: "2024-12-31"
  years: [2020, 2021, 2022, 2023]

# Model settings
model:
  name: "FTW_v1_3_Class_FULL"
  tile_size: 256
  overlap: 32

# Processing
processing:
  composite_method: "median"
  cloud_mask_codes: [0, 1, 3, 7, 8, 9, 10]
```

---

## Expected Outputs

### Data Products

| Output | Format | Location |
|--------|--------|----------|
| Seasonal composites | NetCDF | `data/processed/composites/` |
| Field masks (per year) | NPZ | `data/outputs/masks/` |
| Stability matrix | CSV | `data/outputs/stability_matrix.csv` |
| Change statistics | CSV | `data/outputs/change_stats.csv` |

### Figures

| Figure | Description |
|--------|-------------|
| `stability_heatmap.png` | IoU matrix heatmap |
| `masks_grid.png` | Multi-year field masks |
| `change_2020_2021.png` | Change map example |
| `stability_zones.png` | Temporal stability classification |

---

## Team & Task Allocation

| Phase | Owner | Tasks |
|-------|-------|-------|
| 1: Setup | Ousama Bin Zamir | 1.1-1.4: Data pipeline, STAC, model download |
| 2: Preprocessing | Ahmad Ahmad Abubakar | 2.1-2.4: Cloud mask, composites, NDVI stats |
| 3: Inference | Ousama Bin Zamir | 3.1-3.3: FTW wrapper, batch inference |
| 4: Analysis | Ahmad Ahmad Abubakar | 4.1-4.3: IoU, change detection, stability |
| 5: Visualization | Both | 5.1: Figures and documentation |

---

## References

1. Fields of The World: A Machine Learning Benchmark Dataset For Global Agricultural Field Boundary Segmentation. arXiv:2409.16252 (2024)

2. d'Andrimont, R., et al. AI4Boundaries: An open AI-ready dataset to map field boundaries with Sentinel-2 and aerial photography. Earth System Science Data, 15(1), 317–329 (2023)

3. Watkins, B., & van Niekerk, A. A comparison of object-based image analysis approaches for field boundary delineation using multi-temporal Sentinel-2 imagery. Geospatial Information Science, 22(3), 218–231 (2019)

4. EOPF Zarr Sample Notebooks. https://github.com/EOPF-Sample-Service/eopf-sample-notebooks

---

## License

Apache 2.0

---

## Acknowledgments

- ESA EOPF for Sentinel-2 Zarr data access
- Fields of The World team for pre-trained model weights
- VITO for original parcel delineation notebook inspiration
