"""Validated manual lens correction parameters in oriented source coordinates."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LensCorrections:
    distortion: float = 0.0
    vignetting_ev: float = 0.0

    def __post_init__(self):
        for name, limit in (("distortion", 100), ("vignetting_ev", 2)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not -limit <= value <= limit):
                raise ValueError(f"{name} must be finite and between {-limit} and {limit}.")
            object.__setattr__(self, name, float(value))

    def to_dict(self):
        return {"distortion": self.distortion, "vignetting_ev": self.vignetting_ev}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"distortion", "vignetting_ev"}:
            raise ValueError("Unknown or missing manual lens correction parameters.")
        return cls(**value)


def map_to_source(x, y, size, distortion):
    """Corrected edge coordinates -> original edge coordinates (scalars/arrays).

    The centered safe inset and |k| <= .2 keep the visible radial mapping
    monotone and inside the source rectangle. No clipping hides empty edges.
    """
    if not distortion:
        return x, y
    w, h = size
    radius = math.hypot(w, h) / 2
    k = distortion / 500.
    scale = 1. / (1. + max(0., k))
    u = (x - w/2) / radius * scale
    v = (y - h/2) / radius * scale
    factor = 1. + k * (u*u + v*v)
    return w/2 + radius*u*factor, h/2 + radius*v*factor


def map_from_source(point, size, distortion):
    """Inverse only on the visible monotone ray; omitted handles return None."""
    if not distortion:
        return tuple(point)
    w, h = size
    radius = math.hypot(w, h) / 2
    x, y = (point[0]-w/2)/radius, (point[1]-h/2)/radius
    target = math.hypot(x, y)
    if not target:
        return w/2, h/2
    ux, uy = x/target, y/target
    maximum = min(w/(2*radius*abs(ux)) if ux else math.inf,
                  h/(2*radius*abs(uy)) if uy else math.inf)
    k = distortion/500.
    scale = 1/(1+max(0., k))
    def mapped(r):
        t = r*scale
        return t*(1+k*t*t)
    if target > mapped(maximum) + 1e-13:
        return None
    low, high = 0., maximum
    for _ in range(60):
        mid = (low+high)/2
        if mapped(mid) < target:
            low = mid
        else:
            high = mid
    r = (low+high)/2
    return w/2+radius*ux*r, h/2+radius*uy*r


def apply_vignetting(linear, geometry, ev):
    """Source-radius gain after sampling, before tones; never mutate the source.

    Call on bounded row strips. Positive EV brightens original corners; the
    same floating gain is applied to all channels without clipping highlights.
    """
    if not ev:
        return linear
    import numpy as np
    x, y = geometry.source_grid()
    w, h = geometry.source_size
    r2 = ((x-w/2)**2 + (y-h/2)**2) / ((w*w+h*h)/4)
    np.clip(r2, 0, 1, out=r2)
    r2 *= r2
    r2 *= ev
    np.exp2(r2, out=r2)
    return np.multiply(linear, r2[..., None], dtype=np.float32)
