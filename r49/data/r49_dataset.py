import bisect
from pathlib import Path
from typing import Callable, cast

import torch
from cv2.typing import MatLike
from PIL import Image

from .image_transform import apply_perspective_transform
from .manifest import Manifest
from .r49_file import R49File


class R49Dataset(torch.utils.data.ConcatDataset[tuple[Image.Image, str]]):
    def __init__(
        self,
        r49_files: list[Path],
        *,
        dpt: int = 20,
        size: int = 64,
        labels: list[str] | None = None,
        image_transform: Callable[
            [MatLike, Manifest, int], tuple[MatLike, MatLike]
        ] = apply_perspective_transform,
    ):
        labels = labels if labels is not None else ["track", "train", "other"]
        super().__init__( 
            [
                R49File(r49_file, image_transform=image_transform, size=size, labels=labels)
                for r49_file in r49_files
            ]
        )

    def get_info(self, idx: int) -> tuple[str, int, str]:
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return cast(R49File, self.datasets[dataset_idx]).get_info(sample_idx)
