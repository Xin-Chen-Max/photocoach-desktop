"""Non-destructive crop/straighten geometry in oriented, full-resolution pixels."""
from dataclasses import dataclass
import math
import numpy as np

from .scratch import Scratch


@dataclass(frozen=True)
class Composition:
    # Crop bounds are normalized within the centered, edge-safe rotated canvas.
    crop: tuple[float, float, float, float] = (0., 0., 1., 1.)
    angle: float = 0.0  # clockwise degrees

    def __post_init__(self):
        if (not isinstance(self.crop, (tuple, list)) or len(self.crop) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (float, int)) or
                    not math.isfinite(v) or not 0 <= v <= 1 for v in self.crop)):
            raise ValueError("Crop bounds must be four finite fractions from 0 to 1.")
        l, t, r, b = self.crop
        if r - l < .02 or b - t < .02:
            raise ValueError("Keep at least 2% of each image dimension.")
        if (isinstance(self.angle, bool) or not isinstance(self.angle, (float, int)) or
                not math.isfinite(self.angle) or not -15 <= self.angle <= 15):
            raise ValueError("Straighten must be between -15 and +15 degrees.")
        object.__setattr__(self, "crop", tuple(float(v) for v in self.crop))
        object.__setattr__(self, "angle", float(self.angle))

    def to_dict(self):
        return {"crop": list(self.crop), "angle": self.angle}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"crop", "angle"}:
            raise ValueError("Invalid composition parameters.")
        return cls(value["crop"], value["angle"])


def canvas_size(size, angle):
    """Conservative centered same-aspect canvas with no rotated empty corners."""
    w, h = size
    a = math.radians(abs(angle))
    c, s = math.cos(a), math.sin(a)
    scale = min(w / (c*w + s*h), h / (s*w + c*h))
    return max(1, math.floor(w*scale + 1e-9)), max(1, math.floor(h*scale + 1e-9))


def crop_bounds(size, composition):
    w, h = canvas_size(size, composition.angle)
    l, t, r, b = composition.crop
    left, top = math.floor(l*w), math.floor(t*h)
    return left, top, max(left+1, math.ceil(r*w)), max(top+1, math.ceil(b*h))


def output_size(size, composition):
    l, t, r, b = crop_bounds(size, composition)
    return r-l, b-t


def sample_composition(linear, composition, *, source_size=None, rect=None, target_size=None,
                       scratch: Scratch | None = None):
    """Sample output pixel centers in linear light, including an optional ROI.

    source_size describes the full decode even for reduced preview input. A full
    output and any native ROI use identical inverse coordinates. Identity/crop
    at native size is an exact slice; other transforms use bilinear sampling.
    The caller uses row strips to bound full-resolution temporary allocations.
    """
    linear = np.asarray(linear)
    if (linear.ndim != 3 or linear.shape[2] != 3 or min(linear.shape[:2]) < 1 or
            not np.issubdtype(linear.dtype, np.floating)):
        raise ValueError("Composition requires a nonempty floating-point RGB array.")
    size = source_size or (linear.shape[1], linear.shape[0])
    ow, oh = output_size(size, composition)
    l, t, r, b = rect or (0, 0, ow, oh)
    if not (0 <= l < r <= ow and 0 <= t < b <= oh):
        raise ValueError("Invalid composition output region.")
    tw, th = target_size or (r-l, b-t)
    if tw < 1 or th < 1:
        raise ValueError("Invalid composition sample size.")
    cl, ct, _, _ = crop_bounds(size, composition)
    if composition.angle == 0 and size == (linear.shape[1], linear.shape[0]) and (tw, th) == (r-l, b-t):
        return linear[ct+t:ct+b, cl+l:cl+r]
    if scratch is None:
        scratch = Scratch()
    shape = (th, tw)
    cw, ch = canvas_size(size, composition.angle)
    x = cl + l + (np.arange(tw, dtype=np.float64)+.5)*(r-l)/tw - cw/2
    y = ct + t + (np.arange(th, dtype=np.float64)+.5)*(b-t)/th - ch/2
    a = math.radians(composition.angle)
    c, s = math.cos(a), math.sin(a)
    sx = scratch.get("comp_sx", shape, np.float64)
    sy = scratch.get("comp_sy", shape, np.float64)
    fl = scratch.get("comp_fl", shape, np.float64)
    np.add((c*x)[None, :], (s*y)[:, None], out=sx)
    sx += size[0]/2
    sx *= linear.shape[1]/size[0]
    sx -= .5
    np.clip(sx, 0, linear.shape[1]-1, out=sx)
    np.add((-s*x)[None, :], (c*y)[:, None], out=sy)
    sy += size[1]/2
    sy *= linear.shape[0]/size[1]
    sy -= .5
    np.clip(sy, 0, linear.shape[0]-1, out=sy)
    x0 = scratch.get("comp_x0", shape, np.intp)
    y0 = scratch.get("comp_y0", shape, np.intp)
    x1 = scratch.get("comp_x1", shape, np.intp)
    y1 = scratch.get("comp_y1", shape, np.intp)
    idx = scratch.get("comp_idx", shape, np.intp)
    fx2 = scratch.get("comp_fx", shape, np.float32)
    fy2 = scratch.get("comp_fy", shape, np.float32)
    ofx2 = scratch.get("comp_ofx", shape, np.float32)
    ofy2 = scratch.get("comp_ofy", shape, np.float32)
    np.floor(sx, out=fl)
    np.copyto(x0, fl, casting="unsafe")
    np.subtract(sx, x0, out=fl)
    np.copyto(fx2, fl, casting="unsafe")
    np.floor(sy, out=fl)
    np.copyto(y0, fl, casting="unsafe")
    np.subtract(sy, y0, out=fl)
    np.copyto(fy2, fl, casting="unsafe")
    np.add(x0, 1, out=x1)
    np.minimum(x1, linear.shape[1]-1, out=x1)
    np.add(y0, 1, out=y1)
    np.minimum(y1, linear.shape[0]-1, out=y1)
    np.subtract(np.float32(1), fx2, out=ofx2)
    np.subtract(np.float32(1), fy2, out=ofy2)
    fx = fx2[..., None]
    fy = fy2[..., None]
    ofx = ofx2[..., None]
    ofy = ofy2[..., None]
    stride = linear.shape[1]
    y0 *= stride
    y1 *= stride
    flat = np.ascontiguousarray(linear, dtype=np.float32).reshape(-1, 3)
    rgb_shape = (th, tw, 3)
    out = np.empty(rgb_shape, dtype=np.float32)
    sample = scratch.get("comp_s0", rgb_shape, np.float32)
    lower = scratch.get("comp_s1", rgb_shape, np.float32)
    np.add(y0, x0, out=idx)
    np.take(flat, idx, axis=0, out=out)
    out *= ofx
    np.add(y0, x1, out=idx)
    np.take(flat, idx, axis=0, out=sample)
    sample *= fx
    out += sample
    out *= ofy
    np.add(y1, x0, out=idx)
    np.take(flat, idx, axis=0, out=lower)
    lower *= ofx
    np.add(y1, x1, out=idx)
    np.take(flat, idx, axis=0, out=sample)
    sample *= fx
    lower += sample
    lower *= fy
    out += lower
    return out
