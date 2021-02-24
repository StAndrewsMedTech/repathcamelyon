from pathlib import Path
from typing import List, Tuple

from PIL import Image

from repath.data.slides.slide import SlideBase, Region
from repath.utils.geometry import Size

import repath.data.slides.libraries.pixelengine as pixelengine
import repath.data.slides.libraries.softwarerendercontext as softwarerendercontext
import repath.data.slides.libraries.softwarerenderbackend as softwarerenderbackend

render_context = softwarerendercontext.SoftwareRenderContext()
render_backend = softwarerenderbackend.SoftwareRenderBackend()
pe = pixelengine.PixelEngine(render_backend, render_context)


class Slide(SlideBase):
    # remember there is one of these for each subimage!
    def __init__(self, path: Path, image_index: int = 0) -> None:
        super().__init__()
        self.path = path
        self.image_index = image_index
        self._dims = self._compute_dimensions()
        self.origin = 

    def open(self) -> None:
        self.input = pe['in']
        self.input.open(str(self.path))

    def close(self) -> None:
        self.input.close()

    @property
    def path(self) -> Path:
        return self.path

    @property
    def dimensions(self) -> List[Size]:
        return self._dims

    def read_region(self, region: Region) -> Image:
        level, location, size = region
        range = [location.x, location.x + size.width,
                 location.y, location.y + size.height]
        origin = 
        raise NotImplementedError

    @abstractmethod
    def read_regions(self, regions: List[Region]) -> Image:
        raise NotImplementedError

    def _envelope_rects(self) -> 

    def _compute_dimensions(self) -> List[Size]:
        def get_dims(level: int) -> Size:
            envelopes = self._input.SourceView().dataEnvelopes(level)
            return envelopes.asRectangles()[self.image_index]

        num_levels = self.input.numLevels()
        dims = [get_dims(level) for level in range(num_levels)]
        return dims
