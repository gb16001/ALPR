# mtLPRtr & exDETR

The release of **2 ALPR methods** — a joint License Plate Detection and Recognition model built on ConvNeXt + FPN + STN + Transformer, with an MAE auxiliary task for improved feature representation.

This repository provides:
- **2 model architectures** (mtLPRtr, exDETR)
- **4 pre-trained weight sets** (each model × CCPD / CCPD2018)
- Full training and evaluation code

---

## Models & Weights

| Config | Model | Dataset | Backbone |
|---|---|---|---|
| `CCPD-mtLPRtr_CvNxt_FPN_STN_co2D.yaml` | mtLPRtr | CCPD | ConvNeXt-Tiny |
| `CCPD18-mtLPRtr_CvNxt_FPN_STN_co2D.yaml` | mtLPRtr | CCPD2018 | ConvNeXt-Tiny |
| `CCPD-exDETR_CDN_CvNxt_FPN_STN_co2D.yaml` | exDETR | CCPD | ConvNeXt-Nano |
| `CCPD18-exDETR_CDN_CvNxt_FPN_STN_co2D.yaml` | exDETR | CCPD2018 | ConvNeXt-Nano |

Weights are released as `runs.tar.gz` on the [GitHub Releases](../../releases) page. Extract into the repo root:

```bash
tar -xzvf runs.tar.gz
```

This restores the `runs/` directory that configs already point to.

---

## Installation

Requires Python 3.13.

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Dataset

We do **not** distribute the datasets. Please download them from their official sources:

- **CCPD**: https://github.com/detectRecog/CCPD
- **CCPD2018**: https://github.com/detectRecog/CCPD
- UFPRSR and CTPFSD are also tested. 

After downloading, create symlinks under `datasets/`:

```bash
ln -s /path/to/your/CCPD      datasets/CCPD
ln -s /path/to/your/CCPD2018  datasets/CCPD2018
```

### CSV Format

The dataloader reads a CSV index file. We provide our pre-built CSV files for both datasets (see Releases). Each row describes one image sample:

| Column | Type | Description |
|---|---|---|
| `filename` | str | Image filename |
| `CCPD_path` | str | Relative sub-directory path from the CSV file location |
| `license_plate` | str | License plate string (e.g. `皖A12345`) |
| `bounding_box_1_x` | int | Top-left x of axis-aligned bounding box (pixel) |
| `bounding_box_1_y` | int | Top-left y of axis-aligned bounding box (pixel) |
| `bounding_box_2_x` | int | Bottom-right x of axis-aligned bounding box (pixel) |
| `bounding_box_2_y` | int | Bottom-right y of axis-aligned bounding box (pixel) |
| `vertex_1_x` | int | Plate corner 1 x (pixel) |
| `vertex_1_y` | int | Plate corner 1 y (pixel) |
| `vertex_2_x` | int | Plate corner 2 x (pixel) |
| `vertex_2_y` | int | Plate corner 2 y (pixel) |
| `vertex_3_x` | int | Plate corner 3 x (pixel) |
| `vertex_3_y` | int | Plate corner 3 y (pixel) |
| `vertex_4_x` | int | Plate corner 4 x (pixel) |
| `vertex_4_y` | int | Plate corner 4 y (pixel) |

Example row:

```
filename,CCPD_path,license_plate,bounding_box_1_x,bounding_box_1_y,bounding_box_2_x,bounding_box_2_y,vertex_1_x,vertex_1_y,vertex_2_x,vertex_2_y,vertex_3_x,vertex_3_y,vertex_4_x,vertex_4_y
02-90_85-291&493_415&545-415&540_291&545_291&498_415&493-0_0_3_27_29_27_27_30-109-20.jpg,ccpd_base,皖AD00001,291,493,415,545,415,540,291,545,291,498,415,493
```

---

## Evaluation

```python
from main import eval_Danlu

eval_Danlu('configs/CCPD-mtLPRtr_CvNxt_FPN_STN_co2D.yaml')
```

The config's `CHECKPOINT_DIR` field already points to the correct `runs/` subfolder. Just extract the weights and run.

---

## Training

Edit `cfg_files` in `main.py` to select which configs to train, then call `train_Danlu_scripts()`:

```python
from main import train_Danlu_scripts

cfg_files = [
    'configs/CCPD-mtLPRtr_CvNxt_FPN_STN_co2D.yaml',
    'configs/CCPD18-mtLPRtr_CvNxt_FPN_STN_co2D.yaml',
    # ...
]
train_Danlu_scripts(cfg_list=cfg_files)
```

Checkpoints are saved to the `runs/<name>/` directory defined in each config.

---

