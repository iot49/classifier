import cv2
import numpy as np
from typing import Any, cast

from cv2.typing import MatLike

from .manifest import Manifest


def apply_perspective_transform(
    image: MatLike, manifest: Manifest, dpt: int
) -> tuple[MatLike, MatLike]:
    """Apply perspective transformation to an OpenCV image using calibration data from manifest."""

    calibration = manifest.calibration
    corner_points = ["rect-0", "rect-2", "rect-3", "rect-1"]

    # Source points (the quadrilateral in the image)
    src = np.array(
        [[calibration[key].x, calibration[key].y] for key in corner_points],
        dtype=np.float32,
    )

    # Scaling to meet image resolution target
    layout_size = manifest.layout.size
    scale = dpt / manifest.gauge_mm
    if scale > 3: 
        raise ValueError(f"upscaling image by {scale}x - increase camera resolution!")

    w = scale * (layout_size.width or 0) 
    h = scale * (layout_size.height or 0)

    # Destination points (corners of the output rectangle)
    dst = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]],
        dtype=np.float32,
    )

    # Compute perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(src, dst)

    # Calculate bounds for the entire transformed image to eliminate black borders
    # Transform all corners of the original image to see the full extent
    img_height, img_width = image.shape[:2]  # OpenCV uses (height, width)  # pyright: ignore[reportAny]
    image_corners = np.array(
        [
            [0, 0],
            [img_width, 0],
            [img_width, img_height],
            [0, img_height],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    # Transform the image corners to see the full extent in output space
    c = cv2.perspectiveTransform(image_corners, transform_matrix)
    c = c.reshape(-1, 2)

    # Extract individual corner coordinates
    tl, tr, br, bl = c

    # Calculate exact boundaries to avoid black borders
    # Find the largest rectangle that fits inside the transformed shape
    xl = max(tl[0], bl[0])
    xr = min(tr[0], br[0])
    yt = max(tl[1], tr[1])
    yb = min(bl[1], br[1])

    # Create translation matrix to shift the transformed image to (0,0)
    translation_matrix = np.array(
        [[1, 0, -xl], [0, 1, -yt], [0, 0, 1]], dtype=np.float32
    )

    # Combine original transform with translation
    # Explicitly cast to MatLike to satisfy type checker for cv2 functions
    final_transform = cast(MatLike, translation_matrix @ transform_matrix)

    # Apply perspective transformation with calculated optimal size
    transformed_image = cv2.warpPerspective(
        image,
        final_transform,
        (int(np.ceil(xr - xl)), int(np.ceil(yb - yt))),  # pyright: ignore[reportAny]
        flags=cv2.INTER_CUBIC,
    )

    return (transformed_image, final_transform)
