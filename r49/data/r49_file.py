import json
import zipfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from cv2.typing import MatLike
from PIL import Image

from .image_transform import apply_perspective_transform
from .manifest import Manifest


class R49File(torch.utils.data.Dataset):
    def __init__(
        self,
        r49file: Path,
        *,
        size=64,
        image_transform: Callable[
            [MatLike, Manifest, int], tuple[MatLike, np.ndarray]
        ] = apply_perspective_transform,
        rotation_angles: list[float] = [0],
        dpt: int = 20,
        verbose: bool = False,
    ):
        self._r49file = r49file
        self._size = size
        self._image: MatLike = None  # cv2 image (_read_r49)
        self._image_transform = image_transform
        self._dpt = dpt
        self._rotation_angles = rotation_angles
        self._verbose = verbose

        # read r49 and create all samples
        self._read_r49()
        if self._image is None or self._image.size == 0:
            raise ValueError(f"Failed to load or process image from {self._r49file}")
        self._create_xy()

    @property
    def manifest(self):
        return self._manifest

    @property
    def image(self):
        return self._image

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        # Convert BGR (OpenCV) to tensor ???
        # Could not get this to work for rotation augmentation, so doing rotation in this file instead
        # https://forums.fast.ai/t/opencv-images-np-array-to-fastai-open-image/44468
        # https://github.com/pytorch/vision/issues/8188
        img_cv2_rgb = cv2.cvtColor(self._x[idx], cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_cv2_rgb)
        return pil_img, self._y[idx]

    def save(self, output_path: Path):
        """Save the dataset samples to the specified output path."""
        for i, (img, label) in enumerate(zip(self._x, self._y)):
            label_dir = output_path / label
            label_dir.mkdir(parents=True, exist_ok=True)
            im_file = label_dir / f"{self._r49file.stem}_{i}.jpg"
            cv2.imwrite(str(im_file), img)

    def _read_r49(self):
        with zipfile.ZipFile(self._r49file, "r") as zf:
            with zf.open("manifest.json") as manifest_file:
                manifest_dict = json.load(manifest_file)
                self._manifest = Manifest(**manifest_dict)
                assert self._manifest.version == 1, (
                    f"Got manifest unsupported version {self._manifest.version}. Expected version 1."
                )

            # Find the image file in the zip
            image_files = [
                f
                for f in zf.namelist()
                if f.startswith("image.")
                and f.split(".")[-1].lower() in ["jpeg", "jpg", "png"]
            ]
            if not image_files:
                raise ValueError(f"No image file found in {self._r49file}")

            # Read image bytes from zip and decode with cv2
            image_filename = image_files[0]
            with zf.open(image_filename) as img_file:
                image_bytes = img_file.read()
                # Convert bytes to numpy array and decode with OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                self._image, self.transform = self._image_transform(
                    image=cv2.imdecode(nparr, cv2.IMREAD_COLOR),
                    manifest=self._manifest,
                    dpt=self._dpt,
                )

    def _create_xy(self):
        self._x: list[MatLike] = []
        self._y: list[str] = []
        size = self._size

        for label, ((_, lx), (_, ly)) in self._manifest.markers.label.items():
            label = label.rsplit("-", 1)[0]
            if label in ["train-end", "coupling"]:
                # ignore these for now
                continue

            for angle in self._rotation_angles:
                # Transform the marker position to the transformed image coordinate space
                marker_point = np.array([[[lx, ly]]], dtype=np.float32)
                [[[cx_float, cy_float]]] = cv2.perspectiveTransform(
                    marker_point, self.transform
                )

                cx = int(cx_float)
                cy = int(cy_float)

                # Calculate region size needed for rotation (1.5x crop size suffices for 45deg worst case)
                region_size = int(size * 1.5)
                radius = region_size // 2

                try:
                    # Check bounds by accessing corner pixels - this will raise IndexError if out of bounds
                    _ = self.image[cy - radius, cx - radius]  # top-left
                    _ = self.image[cy + radius - 1, cx + radius - 1]  # bottom-right

                    # Rotate
                    rotated_region = cv2.warpAffine(
                        self.image[
                            cy - radius : cy + radius, cx - radius : cx + radius
                        ],
                        cv2.getRotationMatrix2D((radius, radius), angle, 1.0),
                        (region_size, region_size),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,  # Use constant border (black) instead of reflect
                        borderValue=(0, 0, 0),
                    )

                    cropped_image = rotated_region[
                        radius - size // 2 : radius + size // 2,
                        radius - size // 2 : radius + size // 2,
                    ]

                except (IndexError, cv2.error):
                    print(
                        f"***** WARNING: {self._r49file} marker {label} at {lx}/{ly} angle {angle} too close to image edge, skipping"
                    )
                    if self._verbose:
                        # Create debug visualization
                        img_to_show = self._image.copy()
                        img_h, img_w = self._image.shape[:2]
                        cv2.circle(img_to_show, (int(cx), int(cy)), 5, (0, 0, 255), 2)
                        cv2.rectangle(
                            img_to_show,
                            (max(0, cx - radius), max(0, cy - radius)),
                            (min(img_w, cx + radius), min(img_h, cy + radius)),
                            (255, 0, 0),
                            2,
                        )
                        cv2.imshow("debug_insufficient_space", img_to_show)
                        cv2.waitKey()
                    break

                self._x.append(cropped_image)
                self._y.append(label)

    def __str__(self):
        return f"R49FileDataset(Path('{self._r49file}'))"
