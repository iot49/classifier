from enum import Enum
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

"""
Derived from `layout_ui/src/app/manifest.ts`. Keep in sync!
"""


class ValidScales(str, Enum):
    """Valid model railroad scales."""

    G = "G"
    O = "O"
    S = "S"
    HO = "HO"
    T = "T"
    N = "N"
    Z = "Z"


# Scale to number mapping from TypeScript
SCALE_TO_NUMBER = {
    ValidScales.G: 25,
    ValidScales.O: 48,
    ValidScales.S: 64,
    ValidScales.HO: 87,
    ValidScales.T: 72,
    ValidScales.N: 160,
    ValidScales.Z: 96,
}

STANDARD_GAUGE_MM = 1435  # Standard gauge in millimeters


# Type alias for marker categories (equivalent to TypeScript's MarkerCategory)
MarkerCategory = Literal["calibration", "detector", "label"]


class Marker(BaseModel):
    """A marker point with x,y coordinates."""

    x: int
    y: int


class Size(BaseModel):
    """Layout size dimensions in [mm] as indicated by `markers.calibration`."""

    width: Optional[float] = None
    height: Optional[float] = None


class Layout(BaseModel):
    """Model railroad layout information."""

    scale: ValidScales
    size: Size  # size of markers.calibration in [mm]
    name: Optional[str] = None
    description: Optional[str] = None
    contact: Optional[str] = None


class Resolution(BaseModel):
    """Camera resolution in pixels."""

    width: int
    height: int


class Camera(BaseModel):
    """Camera configuration."""

    resolution: Resolution  # camera resolution, i.e. image size in pixels
    model: Optional[str] = None


class Markers(BaseModel):
    """Collection of markers by category."""

    calibration: Dict[str, Marker] = Field(default_factory=dict)
    label: Dict[str, Marker] = Field(default_factory=dict)
    detector: Dict[str, Marker] = Field(default_factory=dict)


class Manifest(BaseModel):
    """Complete manifest data structure for railroad layout labeling."""

    version: int
    layout: Layout
    camera: Camera
    markers: Markers

    @property
    def get_scale_number(self) -> int:
        """Get the numeric scale value for the layout."""
        return SCALE_TO_NUMBER[self.layout.scale]

    @property
    def gauge_mm(self) -> float:
        """Gauge in mm for the layout scale (~ 16.5mm for HO)."""
        return STANDARD_GAUGE_MM / self.get_scale_number

    @classmethod
    def create_default(cls) -> "Manifest":
        """Create a default manifest with HO scale and empty markers."""
        return cls(
            version=1,
            layout=Layout(
                name=None, scale=ValidScales.HO, size=Size(width=None, height=None)
            ),
            camera=Camera(resolution=Resolution(width=0, height=0)),
            markers=Markers(),
        )
