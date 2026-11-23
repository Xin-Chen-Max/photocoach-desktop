"""Immutable manual gradients and explicit source-coordinate sampling contexts."""
from dataclasses import dataclass, replace
import math
import re
import numpy as np
from .lens import map_to_source, map_from_source
from .composition import Composition, canvas_size, crop_bounds, output_size

MAX_STROKES_PER_MASK = 64
MAX_POINTS_PER_STROKE = 512
MIN_BRUSH_RADIUS = 0.005
MAX_BRUSH_RADIUS = 0.35
DEFAULT_BRUSH_RADIUS = 0.06
DEFAULT_BRUSH_FEATHER = 0.5
MASK_EDGE_RADIUS = 8
MASK_EDGE_EPS = 0.0012
MASK_EDGE_HALO_PX = 16
DEFAULT_LUMA_FEATHER = 0.15
_MASK_GUIDE_KEYS = frozenset(("luma_min", "luma_max", "luma_feather", "edge_aware"))


def number(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Expected a finite number from {low:g} to {high:g}.")
    return float(value)


@dataclass(frozen=True)
class LinearGradient:
    id: str
    name: str
    start: tuple[float, float]
    end: tuple[float, float]
    enabled: bool = True
    inverted: bool = False
    exposure_ev: float = 0.
    highlights: float = 0.
    shadows: float = 0.
    warmth: float = 0.
    luma_min: float = 0.0
    luma_max: float = 1.0
    luma_feather: float = DEFAULT_LUMA_FEATHER
    edge_aware: bool = False

    def _validate_base(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", self.id):
            raise ValueError("Invalid gradient ID.")
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 40 or any(ord(c)<32 for c in self.name):
            raise ValueError("Gradient names require 1–40 printable characters.")
        object.__setattr__(self, "name", self.name.strip())
        if type(self.enabled) is not bool or type(self.inverted) is not bool or type(self.edge_aware) is not bool:
            raise ValueError("Gradient switches must be boolean.")
        for key, limit in (("exposure_ev",3),("highlights",1),("shadows",1),("warmth",1)):
            object.__setattr__(self,key,number(getattr(self,key),-limit,limit))
        for key in ("luma_min", "luma_max", "luma_feather"):
            object.__setattr__(self, key, number(getattr(self, key), 0, 1))
        if self.luma_min > self.luma_max:
            raise ValueError("Luminance range minimum cannot exceed maximum.")

    def __post_init__(self):
        self._validate_base()
        for key in ("start", "end"):
            point = getattr(self, key)
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError("A gradient endpoint needs two coordinates.")
            object.__setattr__(self, key, tuple(number(v,0,1) for v in point))
        if math.dist(self.start, self.end) < 1e-5:
            raise ValueError("Drag farther to create a gradient transition.")

    @property
    def active(self):
        return self.enabled and bool(self.exposure_ev or self.highlights or self.shadows or self.warmth)

    @property
    def has_luma_range(self):
        return self.luma_min > 0.0 or self.luma_max < 1.0

    @property
    def uses_image_guide(self):
        return self.has_luma_range or self.edge_aware

    def _guide_dict(self):
        extra = {}
        if self.has_luma_range or self.luma_feather != DEFAULT_LUMA_FEATHER:
            extra["luma_min"] = self.luma_min
            extra["luma_max"] = self.luma_max
            extra["luma_feather"] = self.luma_feather
        if self.edge_aware:
            extra["edge_aware"] = True
        return extra

    def to_dict(self):
        return dict(id=self.id,name=self.name,start=list(self.start),end=list(self.end),
                    enabled=self.enabled,inverted=self.inverted,exposure_ev=self.exposure_ev,
                    highlights=self.highlights,shadows=self.shadows,
                    **({"warmth":self.warmth} if self.warmth else {}),
                    **self._guide_dict())

    @classmethod
    def from_dict(cls,value):
        base = {"id","name","start","end","enabled","inverted","exposure_ev","highlights","shadows"}
        if not isinstance(value,dict) or not (base <= set(value) <= (base | {"warmth"} | _MASK_GUIDE_KEYS)):
            raise ValueError("Invalid gradient parameters.")
        return cls(**value)


@dataclass(frozen=True)
class RadialGradient(LinearGradient):
    """An ellipse whose opposite bounding corners use original-image coordinates."""
    feather: float = .5
    angle: float = 0.

    def __post_init__(self):
        super().__post_init__()
        if any(abs(a-b)<1e-5 for a,b in zip(self.start,self.end)):
            raise ValueError("Drag wider and taller to create a radial mask.")
        object.__setattr__(self,'feather',number(self.feather,0,1))
        object.__setattr__(self,'angle',number(self.angle,-90,90))

    def to_dict(self):
        d = dict(super().to_dict(),kind='radial',feather=self.feather)
        if self.angle:
            d['angle'] = self.angle
        return d

    @classmethod
    def from_dict(cls,value):
        required={'id','name','start','end','enabled','inverted','exposure_ev','highlights','shadows','kind','feather'}
        if not isinstance(value,dict) or not (required <= set(value) <= (required | {'warmth','angle'} | _MASK_GUIDE_KEYS)) or value.get('kind')!='radial':
            raise ValueError('Invalid radial mask parameters.')
        return cls(**{k:v for k,v in value.items() if k!='kind'})


@dataclass(frozen=True)
class BrushStroke:
    """One bounded paint or erase polyline in original-oriented [0, 1] coordinates."""
    points: tuple[tuple[float, float], ...]
    radius: float = DEFAULT_BRUSH_RADIUS
    feather: float = DEFAULT_BRUSH_FEATHER
    erase: bool = False

    def __post_init__(self):
        if not isinstance(self.points, (tuple, list)) or not 1 <= len(self.points) <= MAX_POINTS_PER_STROKE:
            raise ValueError(f"A brush stroke requires 1–{MAX_POINTS_PER_STROKE} points.")
        normalized = []
        for point in self.points:
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError("Each brush stroke point needs two coordinates.")
            normalized.append((number(point[0], 0, 1), number(point[1], 0, 1)))
        object.__setattr__(self, "points", tuple(normalized))
        object.__setattr__(self, "radius", number(self.radius, MIN_BRUSH_RADIUS, MAX_BRUSH_RADIUS))
        object.__setattr__(self, "feather", number(self.feather, 0, 1))
        if type(self.erase) is not bool:
            raise ValueError("Brush erase flag must be boolean.")

    def to_dict(self):
        return {
            "points": [[x, y] for x, y in self.points],
            "radius": self.radius,
            "feather": self.feather,
            "erase": self.erase,
        }

    @classmethod
    def from_dict(cls, value):
        required = {"points", "radius", "feather", "erase"}
        if (not isinstance(value, dict) or set(value) != required
                or not isinstance(value["points"], list)
                or not all(isinstance(p, list) for p in value["points"])):
            raise ValueError("Invalid brush stroke parameters.")
        return cls(**value)


@dataclass(frozen=True)
class BrushMask(LinearGradient):
    """Non-destructive painted/erased mask in original-oriented normalized coordinates."""
    start: tuple[float, float] = (0.0, 0.0)
    end: tuple[float, float] = (1.0, 1.0)
    strokes: tuple[BrushStroke, ...] = ()

    def __post_init__(self):
        self._validate_base()
        if (not isinstance(self.strokes, tuple)
                or not 1 <= len(self.strokes) <= MAX_STROKES_PER_MASK
                or any(not isinstance(s, BrushStroke) for s in self.strokes)):
            raise ValueError(f"A brush mask requires 1–{MAX_STROKES_PER_MASK} valid brush strokes.")
        xs = [p[0] for stroke in self.strokes for p in stroke.points]
        ys = [p[1] for stroke in self.strokes for p in stroke.points]
        object.__setattr__(self, "start", (min(xs), min(ys)))
        object.__setattr__(self, "end", (max(xs), max(ys)))

    def to_dict(self):
        return dict(
            id=self.id, name=self.name, kind="brush",
            strokes=[s.to_dict() for s in self.strokes],
            enabled=self.enabled, inverted=self.inverted,
            exposure_ev=self.exposure_ev, highlights=self.highlights,
            shadows=self.shadows,
            **({"warmth": self.warmth} if self.warmth else {}),
            **self._guide_dict(),
        )

    @classmethod
    def from_dict(cls, value):
        required = {"id", "name", "kind", "strokes", "enabled", "inverted", "exposure_ev", "highlights", "shadows"}
        if (not isinstance(value, dict)
                or not (required <= set(value) <= (required | {"warmth"} | _MASK_GUIDE_KEYS))
                or value.get("kind") != "brush"
                or not isinstance(value.get("strokes"), list)):
            raise ValueError("Invalid brush mask parameters.")
        strokes = tuple(BrushStroke.from_dict(s) for s in value["strokes"])
        fields = {k: v for k, v in value.items() if k not in ("kind", "strokes")}
        return cls(**fields, strokes=strokes)


def gradient_from_dict(value):
    if isinstance(value,dict) and value.get('kind')=='radial':
        return RadialGradient.from_dict(value)
    if isinstance(value,dict) and value.get('kind')=='brush':
        return BrushMask.from_dict(value)
    return LinearGradient.from_dict(value)


def validate_gradients(items):
    if not isinstance(items,tuple) or len(items)>4 or any(not isinstance(g,LinearGradient) for g in items):
        raise ValueError("Use up to four immutable local masks.")
    if len({g.id for g in items})!=len(items) or len({g.name.casefold() for g in items})!=len(items):
        raise ValueError("Gradient IDs and names must be unique.")
    return items


@dataclass(frozen=True)
class RenderGeometry:
    source_size: tuple[int,int]
    composition: Composition
    rect: tuple[float,float,float,float]
    target_size: tuple[int,int]
    distortion: float = 0.

    def __post_init__(self):
        if len(self.source_size)!=2 or any(type(v) is not int or v<1 for v in self.source_size):
            raise ValueError("Invalid original image size.")
        if len(self.target_size)!=2 or any(type(v) is not int or v<1 for v in self.target_size):
            raise ValueError("Invalid sample grid size.")
        w,h=output_size(self.source_size,self.composition)
        if len(self.rect)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in self.rect):
            raise ValueError("Invalid sample region.")
        l,t,r,b=self.rect
        if not (0<=l<r<=w and 0<=t<b<=h):
            raise ValueError("Sample region is outside the photo.")

    @classmethod
    def full(cls,source_size,composition,target_size,distortion=0.):
        return cls(tuple(source_size),composition,(0,0,*output_size(source_size,composition)),tuple(target_size),distortion)

    def rows(self,top,bottom):
        l,t,r,b=self.rect
        height=self.target_size[1]
        return replace(self,rect=(l,t+(b-t)*top/height,r,t+(b-t)*bottom/height),
                       target_size=(self.target_size[0],bottom-top))

    def source_grid(self):
        """Original pixel-center coordinates, before decoder-resolution scaling."""
        l,t,r,b=self.rect; tw,th=self.target_size
        cl,ct,_,_=crop_bounds(self.source_size,self.composition)
        cw,ch=canvas_size(self.source_size,self.composition.angle)
        x=cl+l+(np.arange(tw,dtype=np.float64)+.5)*(r-l)/tw-cw/2
        y=ct+t+(np.arange(th,dtype=np.float64)+.5)*(b-t)/th-ch/2
        a=math.radians(self.composition.angle); c,s=math.cos(a),math.sin(a)
        return map_to_source(c*x[None,:]+s*y[:,None]+self.source_size[0]/2,
                -s*x[None,:]+c*y[:,None]+self.source_size[1]/2,self.source_size,self.distortion)


def source_point(point,source_size,composition,*,inverse=False,distortion=0.):
    """Map a composed canvas pixel center to original pixels, or reverse it."""
    cw,ch=canvas_size(source_size,composition.angle)
    cl,ct,_,_=crop_bounds(source_size,composition)
    a=math.radians(composition.angle); c,s=math.cos(a),math.sin(a)
    if inverse:
        point=map_from_source(point,source_size,distortion)
        if point is None:
            return None
        x,y=point[0]-source_size[0]/2,point[1]-source_size[1]/2
        return (c*x-s*y+cw/2-cl,s*x+c*y+ch/2-ct)
    x,y=point[0]+cl-cw/2,point[1]+ct-ch/2
    return map_to_source(c*x+s*y+source_size[0]/2,-s*x+c*y+source_size[1]/2,source_size,distortion)


def _brush_weight_strip(gradient: BrushMask, sx: np.ndarray, sy: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    """Evaluate a BrushMask on one bounded row strip in original-pixel coordinates."""
    w, h = source_size
    ref = float(min(w, h))
    sx_f = np.asarray(sx, dtype=np.float32)
    sy_f = np.asarray(sy, dtype=np.float32)
    sx_min = float(sx_f.min())
    sx_max = float(sx_f.max())
    sy_min = float(sy_f.min())
    sy_max = float(sy_f.max())
    weight = np.zeros(sx_f.shape, dtype=np.float32)
    for stroke in gradient.strokes:
        r_px = np.float32(max(1e-6, stroke.radius * ref))
        r_px2 = np.float32(r_px * r_px)
        pad = float(r_px) + 1e-3
        px = [p[0] * w for p in stroke.points]
        py = [p[1] * h for p in stroke.points]
        x0, x1 = min(px) - pad, max(px) + pad
        y0, y1 = min(py) - pad, max(py) + pad
        if sx_max < x0 or sx_min > x1 or sy_max < y0 or sy_min > y1:
            continue
        mask_box = (
            (sx_f >= np.float32(x0))
            & (sx_f <= np.float32(x1))
            & (sy_f >= np.float32(y0))
            & (sy_f <= np.float32(y1))
        )
        rows_any = np.any(mask_box, axis=1)
        if not rows_any.any():
            continue
        cols_any = np.any(mask_box, axis=0)
        r_lo = int(np.argmax(rows_any))
        r_hi = int(len(rows_any) - np.argmax(rows_any[::-1]))
        c_lo = int(np.argmax(cols_any))
        c_hi = int(len(cols_any) - np.argmax(cols_any[::-1]))
        sub_sx = sx_f[r_lo:r_hi, c_lo:c_hi]
        sub_sy = sy_f[r_lo:r_hi, c_lo:c_hi]
        sub_sx_min = float(sub_sx.min())
        sub_sx_max = float(sub_sx.max())
        sub_sy_min = float(sub_sy.min())
        sub_sy_max = float(sub_sy.max())
        min_d2 = np.full(sub_sx.shape, np.inf, dtype=np.float32)
        dx = np.empty(sub_sx.shape, dtype=np.float32)
        dy = np.empty(sub_sx.shape, dtype=np.float32)
        t_buf = np.empty(sub_sx.shape, dtype=np.float32)
        d2 = np.empty(sub_sx.shape, dtype=np.float32)
        if len(px) == 1:
            ax, ay = np.float32(px[0]), np.float32(py[0])
            np.subtract(sub_sx, ax, out=dx)
            np.subtract(sub_sy, ay, out=dy)
            np.multiply(dx, dx, out=d2)
            np.multiply(dy, dy, out=t_buf)
            d2 += t_buf
            np.minimum(min_d2, d2, out=min_d2)
        else:
            for i in range(len(px) - 1):
                ax_f, ay_f = px[i], py[i]
                bx_f, by_f = px[i + 1], py[i + 1]
                if (sub_sx_max < min(ax_f, bx_f) - pad or sub_sx_min > max(ax_f, bx_f) + pad
                        or sub_sy_max < min(ay_f, by_f) - pad or sub_sy_min > max(ay_f, by_f) + pad):
                    continue
                ax, ay = np.float32(ax_f), np.float32(ay_f)
                bx, by = np.float32(bx_f), np.float32(by_f)
                vx, vy = np.float32(bx - ax), np.float32(by - ay)
                len2 = float(vx) * float(vx) + float(vy) * float(vy)
                np.subtract(sub_sx, ax, out=dx)
                np.subtract(sub_sy, ay, out=dy)
                if len2 > 1e-12:
                    np.multiply(dx, vx, out=t_buf)
                    np.multiply(dy, vy, out=d2)
                    t_buf += d2
                    t_buf *= np.float32(1.0 / len2)
                    np.clip(t_buf, np.float32(0.0), np.float32(1.0), out=t_buf)
                    np.multiply(t_buf, vx, out=d2)
                    dx -= d2
                    np.multiply(t_buf, vy, out=d2)
                    dy -= d2
                np.multiply(dx, dx, out=d2)
                np.multiply(dy, dy, out=t_buf)
                d2 += t_buf
                np.minimum(min_d2, d2, out=min_d2)
        if stroke.feather > 0.0:
            outside = min_d2 >= r_px2
            np.minimum(min_d2, r_px2, out=min_d2)
            np.sqrt(min_d2, out=min_d2)
            min_d2 /= r_px
            inner = np.float32(1.0 - stroke.feather)
            inv_f = np.float32(1.0 / stroke.feather)
            min_d2 -= inner
            min_d2 *= inv_f
            np.clip(min_d2, np.float32(0.0), np.float32(1.0), out=min_d2)
            np.multiply(min_d2, np.float32(-2.0), out=t_buf)
            t_buf += np.float32(3.0)
            np.multiply(min_d2, min_d2, out=d2)
            d2 *= t_buf
            np.subtract(np.float32(1.0), d2, out=d2)
            d2[outside] = np.float32(0.0)
            alpha = d2
        else:
            alpha = (min_d2 <= r_px2).astype(np.float32)
        sub_w = weight[r_lo:r_hi, c_lo:c_hi]
        if stroke.erase:
            np.subtract(np.float32(1.0), alpha, out=t_buf)
            sub_w *= t_buf
        else:
            np.maximum(sub_w, alpha, out=sub_w)
    if gradient.inverted:
        np.subtract(np.float32(1.0), weight, out=weight)
    return weight


def gradient_weight(gradient,geometry,grid=None):
    sx,sy=geometry.source_grid() if grid is None else grid
    w,h=geometry.source_size
    if isinstance(gradient,BrushMask):
        if sx.shape[0]<=128:
            return _brush_weight_strip(gradient,sx,sy,(w,h))
        weight=np.empty(sx.shape,dtype=np.float32)
        for top in range(0,sx.shape[0],128):
            bottom=min(sx.shape[0],top+128)
            weight[top:bottom]=_brush_weight_strip(gradient,sx[top:bottom],sy[top:bottom],(w,h))
        return weight
    if isinstance(gradient,RadialGradient):
        cx=(gradient.start[0]+gradient.end[0])*w/2
        cy=(gradient.start[1]+gradient.end[1])*h/2
        rx=abs(gradient.end[0]-gradient.start[0])*w/2
        ry=abs(gradient.end[1]-gradient.start[1])*h/2
        dx,dy=sx-cx,sy-cy
        if gradient.angle:
            rad=math.radians(gradient.angle)
            ca,sa=math.cos(rad),math.sin(rad)
            dx,dy=ca*dx+sa*dy,-sa*dx+ca*dy
        radius=np.sqrt((dx/rx)**2+(dy/ry)**2)
        if gradient.feather:
            t=np.clip((radius-(1-gradient.feather))/gradient.feather,0,1).astype(np.float32)
            weight=np.clip(
                np.float32(1)-t*t*t*(t*(np.float32(6)*t-np.float32(15))+np.float32(10)),
                np.float32(0),
                np.float32(1),
            )
        else:
            weight=(radius<=1).astype(np.float32)
        return np.float32(1)-weight if gradient.inverted else weight
    ax,ay=gradient.start[0]*w,gradient.start[1]*h
    dx,dy=(gradient.end[0]-gradient.start[0])*w,(gradient.end[1]-gradient.start[1])*h
    t=np.clip(((sx-ax)*dx+(sy-ay)*dy)/(dx*dx+dy*dy),0,1).astype(np.float32)
    t=t*t*(np.float32(3)-np.float32(2)*t)
    return t if gradient.inverted else np.float32(1)-t


def perceptual_luma_guide(linear_rgb: np.ndarray, global_ev: float = 0.0) -> np.ndarray:
    """Approximate perceptual luminance [0, 1] in linear-light working space for mask gating."""
    scale = np.float32(2.0 ** global_ev)
    y = (
        np.float32(0.2126) * linear_rgb[..., 0]
        + np.float32(0.7152) * linear_rgb[..., 1]
        + np.float32(0.0722) * linear_rgb[..., 2]
    ) * scale
    np.clip(y, np.float32(0.0), np.float32(1.0), out=y)
    return np.power(y, np.float32(1.0 / 2.2)).astype(np.float32, copy=False)


def _luma_range_weight(guide_l: np.ndarray, luma_min: float, luma_max: float, luma_feather: float) -> np.ndarray:
    """Smooth perceptual luminance gating window in [0, 1]."""
    out = np.ones(guide_l.shape, dtype=np.float32)
    feather = float(luma_feather)
    if luma_min > 0.0:
        if feather > 1e-6:
            t = np.clip((guide_l - np.float32(luma_min - feather)) / np.float32(feather), np.float32(0.0), np.float32(1.0))
            out *= t * t * (np.float32(3.0) - np.float32(2.0) * t)
        else:
            out *= (guide_l >= np.float32(luma_min)).astype(np.float32)
    if luma_max < 1.0:
        if feather > 1e-6:
            t = np.clip((np.float32(luma_max + feather) - guide_l) / np.float32(feather), np.float32(0.0), np.float32(1.0))
            out *= t * t * (np.float32(3.0) - np.float32(2.0) * t)
        else:
            out *= (guide_l <= np.float32(luma_max)).astype(np.float32)
    return out


def _guided_filter_mask(
    weight: np.ndarray,
    guide_l: np.ndarray,
    radius: int = MASK_EDGE_RADIUS,
    eps: float = MASK_EDGE_EPS,
) -> np.ndarray:
    """Edge-aware mask refinement guided by perceptual image luminance."""
    from .detail import _box_mean_stack
    guide = np.asarray(guide_l, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    stack = np.stack((guide, w, guide * w, guide * guide), axis=0)
    mean_i, mean_p, mean_ip, mean_ii = _box_mean_stack(stack, radius)
    var_i = np.maximum(mean_ii - mean_i * mean_i, np.float32(0.0))
    cov_ip = mean_ip - mean_i * mean_p
    a = cov_ip / (var_i + np.float32(eps))
    b = mean_p - a * mean_i
    mean_a, mean_b = _box_mean_stack(np.stack((a, b), axis=0), radius)
    return np.clip(mean_a * guide + mean_b, np.float32(0.0), np.float32(1.0)).astype(np.float32, copy=False)


def effective_mask_weight(gradient, geometry, grid=None, guide_l=None) -> np.ndarray:
    """Evaluate spatial geometry plus optional luminance range and edge-aware snapping."""
    weight = gradient_weight(gradient, geometry, grid)
    if guide_l is not None:
        if gradient.has_luma_range:
            weight = (weight * _luma_range_weight(guide_l, gradient.luma_min, gradient.luma_max, gradient.luma_feather)).astype(np.float32, copy=False)
        if gradient.edge_aware:
            weight = _guided_filter_mask(weight, guide_l)
    return weight


def local_mask_fields(gradients, geometry, highlights, shadows, grid=None, guide_l=None):
    """Evaluate tone fields and warmth field together so each active mask runs once."""
    grid = geometry.source_grid() if grid is None else grid
    shape = (geometry.target_size[1], geometry.target_size[0])
    ev = np.zeros(shape, np.float32)
    hl = np.full(shape, highlights, np.float32)
    sh = np.full(shape, shadows, np.float32)
    wf = None
    for g in gradients:
        if g.active:
            weight = effective_mask_weight(g, geometry, grid, guide_l=guide_l)
            if g.exposure_ev:
                ev += weight * np.float32(g.exposure_ev)
            if g.highlights:
                hl += weight * np.float32(g.highlights)
            if g.shadows:
                sh += weight * np.float32(g.shadows)
            if g.warmth:
                if wf is None:
                    wf = np.zeros(shape, np.float32)
                wf += weight * np.float32(g.warmth)
    np.clip(hl, -1, 1, out=hl)
    np.clip(sh, -1, 1, out=sh)
    if wf is not None:
        np.clip(wf, -1.5, 1.5, out=wf)
    return (ev, hl, sh), wf


def tone_fields(gradients,geometry,highlights,shadows,grid=None,guide_l=None):
    grid=geometry.source_grid() if grid is None else grid
    shape=(geometry.target_size[1],geometry.target_size[0])
    ev=np.zeros(shape,np.float32)
    hl=np.full(shape,highlights,np.float32); sh=np.full(shape,shadows,np.float32)
    for g in gradients:
        if g.active:
            weight=effective_mask_weight(g,geometry,grid,guide_l=guide_l)
            ev+=weight*np.float32(g.exposure_ev)
            hl+=weight*np.float32(g.highlights)
            sh+=weight*np.float32(g.shadows)
    np.clip(hl,-1,1,out=hl); np.clip(sh,-1,1,out=sh)
    return ev,hl,sh


def warmth_field(gradients,geometry,grid=None,guide_l=None):
    if not any(g.active and g.warmth for g in gradients):
        return None
    grid=geometry.source_grid() if grid is None else grid
    shape=(geometry.target_size[1],geometry.target_size[0])
    wf=np.zeros(shape,np.float32)
    for g in gradients:
        if g.active and g.warmth:
            wf+=effective_mask_weight(g,geometry,grid,guide_l=guide_l)*np.float32(g.warmth)
    np.clip(wf,-1.5,1.5,out=wf)
    return wf


@dataclass(frozen=True)
class OverlaySpec:
    mode: str
    gradient: LinearGradient | None = None


def overlay_pixels(pixels,geometry,spec):
    """One bounded display-only RGBA overlay; never part of image processing."""
    if spec is None:
        return None
    if pixels.nbytes*7/3+1600*1600*3>64*1024**2:
        raise MemoryError("Overlay exceeds display memory budget; zoom in or resize the window.")
    h,w=pixels.shape[:2]
    result=np.zeros((h,w,4),np.uint8)
    uses_guide = bool(spec.mode == "mask" and spec.gradient and spec.gradient.uses_image_guide)
    edge_halo = MASK_EDGE_HALO_PX if (uses_guide and spec.gradient.edge_aware) else 0
    for top in range(0,h,128):
        bottom=min(h,top+128); block=result[top:bottom]
        if spec.mode=="clipping":
            image=pixels[top:bottom]
            block[np.any(image==255,axis=2)]=(255,70,70,170)
            block[np.all(image==0,axis=2)]=(65,115,255,170)
        elif spec.mode=="mask" and spec.gradient:
            if uses_guide:
                p_top = max(0, top - edge_halo)
                p_bottom = min(h, bottom + edge_halo)
                rgb_strip = pixels[p_top:p_bottom].astype(np.float32) / np.float32(255.0)
                guide_strip = (
                    np.float32(0.2126) * rgb_strip[..., 0]
                    + np.float32(0.7152) * rgb_strip[..., 1]
                    + np.float32(0.0722) * rgb_strip[..., 2]
                ).astype(np.float32, copy=False)
                padded_w = effective_mask_weight(spec.gradient, geometry.rows(p_top, p_bottom), guide_l=guide_strip)
                weight = padded_w[top - p_top : top - p_top + (bottom - top)]
            else:
                weight=gradient_weight(spec.gradient,geometry.rows(top,bottom))
            block[...,:3]=(70,205,210)
            block[...,3]=np.rint(weight*90).astype(np.uint8)
    result.setflags(write=False)
    return result
