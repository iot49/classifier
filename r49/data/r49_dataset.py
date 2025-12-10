from pathlib import Path
from typing import Callable

import numpy as np
import torch
from cv2.typing import MatLike

from .image_transform import apply_perspective_transform
from .manifest import Manifest
from .r49_file import R49File


class R49Dataset(torch.utils.data.ConcatDataset):
    def __init__(
        self,
        r49_files: list[Path],
        image_transform: Callable[
            [MatLike, Manifest, int], tuple[MatLike, np.ndarray]
        ] = apply_perspective_transform,
        size: int = 64,
    ):
        super().__init__(
            [
                R49File(r49_file, image_transform=image_transform, size=size)
                for r49_file in r49_files
            ]
        )
