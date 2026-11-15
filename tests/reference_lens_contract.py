"""Scalar lens-spec oracle for preparation, NOT an application processing path.

The production lens feature is gated on P0 acceptance. This deliberately slow
oracle defines coordinates/EV semantics for future vectorized implementation.
Passing its checks establishes the reference contract, not feature acceptance.
"""
import math


def safe_scale(distortion):
    """Distortion is [-100, 100]; k is [-.2, .2]. Conservative safe inset."""
    k = distortion / 500.0
    return 1.0 / (1.0 + max(0.0, k))


def canvas_to_source(point, size, distortion):
    """Edge coordinates: center W/2,H/2; pixel (i,j) has center i+.5,j+.5."""
    if distortion == 0:
        return tuple(point)
    w, h = size
    radius = math.hypot(w, h) / 2.0
    x, y = (point[0] - w/2.0) / radius, (point[1] - h/2.0) / radius
    scale = safe_scale(distortion)
    x, y = x * scale, y * scale
    factor = 1.0 + (distortion / 500.0) * (x*x + y*y)
    return w/2.0 + radius*x*factor, h/2.0 + radius*y*factor


def source_to_canvas(point, size, distortion):
    """Return None for a source handle outside the corrected canvas footprint.

    Solve only on the monotone visible ray; extending the negative-k polynomial
    beyond the canvas can fold and would invent a misleading handle location.
    """
    if distortion == 0:
        return tuple(point)
    w, h = size
    half_diagonal = math.hypot(w, h) / 2.0
    x, y = (point[0] - w/2.0) / half_diagonal, (point[1] - h/2.0) / half_diagonal
    target = math.hypot(x, y)
    if target == 0:
        return w/2.0, h/2.0
    ux, uy = x/target, y/target
    maximum = min(w/(2.0*half_diagonal*abs(ux)) if ux else math.inf,
                  h/(2.0*half_diagonal*abs(uy)) if uy else math.inf)
    scale, k = safe_scale(distortion), distortion/500.0
    def mapped(radius):
        t = radius*scale
        return t*(1.0+k*t*t)
    if target > mapped(maximum) + 1e-13:
        return None
    low, high = 0.0, maximum
    for _ in range(60):
        middle = (low+high)/2.0
        if mapped(middle) < target:
            low = middle
        else:
            high = middle
    radius = (low+high)/2.0
    return w/2.0+half_diagonal*ux*radius, h/2.0+half_diagonal*uy*radius


def vignette_gain(point, size, corner_ev):
    """Gain at original source coordinates; +EV brightens, same gain for RGB."""
    if corner_ev == 0:
        return 1.0
    w, h = size
    radius2 = ((point[0]-w/2.0)**2+(point[1]-h/2.0)**2)/((w*w+h*h)/4.0)
    return 2.0 ** (corner_ev * min(1.0, max(0.0, radius2))**2)


def sample_reference(linear, point, corner_ev=0, *, source_size=None):
    """One float bilinear interpolation, then source-radius linear-light gain.

    Clamp sample indices only for the ordinary half-pixel edge extension; do not
    use this to disguise out-of-bounds lens geometry. The zero production path
    must continue to use the existing exact-slice/render implementation.
    """
    h, w, _ = linear.shape
    sw, sh = source_size or (w, h)
    x = min(w-1.0, max(0.0, point[0]*w/sw-.5))
    y = min(h-1.0, max(0.0, point[1]*h/sh-.5))
    left, top = math.floor(x), math.floor(y)
    right, bottom = min(left+1, w-1), min(top+1, h-1)
    fx, fy = x-left, y-top
    gain = vignette_gain(point, (sw, sh), corner_ev)
    return tuple(float(((linear[top,left,c]*(1-fx)+linear[top,right,c]*fx)*(1-fy)
                        +(linear[bottom,left,c]*(1-fx)+linear[bottom,right,c]*fx)*fy)*gain)
                 for c in range(3))
