#!/usr/bin/env python3

# ruff: noqa: F403, F405

from pathlib import Path

import matplotlib.pyplot as plt
from fastai.vision.all import *

from r49.data import R49File


def check_gpu():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")

CREATE_DB = True
SHOW_IMG_SAMPLES = True
LEARN = False

DATA_DIR = Path("data")
DPT = 20
SIZE = 96

DB_DIR = Path("../datasets/train-track")
R49_DIR = DB_DIR / "r49"
SAMPLES_DIR = DB_DIR / "samples"


def r49_transforms(
    mult: float = 1.0,  # Multiplication applying to `max_rotate`,`max_lighting`,`max_warp`
    do_flip: bool = True,  # Random flipping
    flip_vert: bool = False,  # Flip vertically
    max_rotate: float = 10.0,  # Maximum degree of rotation
    min_zoom: float = 1.0,  # Minimum zoom
    max_zoom: float = 1.1,  # Maximum zoom
    max_lighting: float = 0.2,  # Maximum scale of changing brightness
    max_warp: float = 0.2,  # Maximum value of changing warp per
    p_affine: float = 0.75,  # Probability of applying affine transformation
    p_lighting: float = 0.75,  # Probability of changing brightnest and contrast
    xtra_tfms: list = None,  # Custom Transformations
    size: int | tuple = None,  # Output size, duplicated if one value is specified
    mode: str = "bilinear",  # PyTorch `F.grid_sample` interpolation
    pad_mode=PadMode.Reflection,  # A `PadMode`
    align_corners=True,  # PyTorch `F.grid_sample` align_corners
    batch=False,  # Apply identical transformation to entire batch
    min_scale=1.0,  # Minimum scale of the crop, in relation to image area
):
    "Utility func to easily create a list of flip, rotate, zoom, warp, lighting transforms."
    res, tkw = (
        [],
        dict(
            size=size if min_scale == 1.0 else None,
            mode=mode,
            pad_mode=pad_mode,
            batch=batch,
            align_corners=align_corners,
        ),
    )
    max_rotate, max_lighting, max_warp = (
        array([max_rotate, max_lighting, max_warp]) * mult
    )
    if do_flip:
        res.append(Dihedral(p=0.5, **tkw) if flip_vert else Flip(p=0.5, **tkw))
    if max_warp:
        res.append(Warp(magnitude=max_warp, p=p_affine, **tkw))
    if max_rotate:
        res.append(Rotate(max_deg=max_rotate, p=p_affine, **tkw))
    if min_zoom < 1 or max_zoom > 1:
        res.append(Zoom(min_zoom=min_zoom, max_zoom=max_zoom, p=p_affine, **tkw))
    if max_lighting:
        res.append(Brightness(max_lighting=max_lighting, p=p_lighting, batch=batch))
        res.append(Contrast(max_lighting=max_lighting, p=p_lighting, batch=batch))
    if min_scale != 1.0:
        xtra_tfms = RandomResizedCropGPU(size, min_scale=min_scale, ratio=(1, 1)) + L(
            xtra_tfms
        )
    return setup_aug_tfms(res + L(xtra_tfms))


# create db from R49 files
if CREATE_DB:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    for r49_file in DATA_DIR.rglob("**/*.r49"):
        r49 = R49File(r49_file, size=64, rotation_angles=[0], dpt=20, verbose=False)
        print(f"Created {len(r49):3} samples from {r49_file}")
        r49.save(DB_DIR)

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=RandomSplitter(valid_pct=0.5, seed=42),
    get_items=get_image_files,
    get_y=parent_label,
)
dblock = dblock.new(
    # item_tfms=Resize(SIZE),
    batch_tfms=r49_transforms(
        mult=1.0,
        max_rotate=180.0,
        max_lighting=0.2,
        max_warp=0,
        size=SIZE,
    )
)
dls = dblock.dataloaders(DB_DIR, bs=32)

# Show some samples from the training set
if SHOW_IMG_SAMPLES:
    dls.train.show_batch(max_n=24, figsize=(9, 9))
    plt.show()


if LEARN:
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(1)

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(9, figsize=(9, 9))

    learn.show_results()
    plt.show()
