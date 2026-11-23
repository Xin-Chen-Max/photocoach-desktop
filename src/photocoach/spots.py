"""Non-destructive circular spot removal (heal & clone) in original-oriented source coordinates."""
from __future__ import annotations

from concurrent.futures import CancelledError
from dataclasses import dataclass
import math
import re
import numpy as np

from .gradients import RenderGeometry

MIN_SPOT_RADIUS = 0.003
MAX_SPOT_RADIUS = 0.15
DEFAULT_SPOT_RADIUS = 0.025
DEFAULT_SPOT_FEATHER = 0.5
DEFAULT_SPOT_OPACITY = 1.0
MAX_SPOTS = 64
SPOT_MODES = ("heal", "clone")

_HEAL_EPS = np.float32(0.01)
_RING_ANGLES = 32
_THETA = np.linspace(0.0, 2.0 * math.pi, _RING_ANGLES, endpoint=False, dtype=np.float64)
_COS_THETA = np.cos(_THETA)
_SIN_THETA = np.sin(_THETA)


def _finite_number(value, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number from {low:g} to {high:g}.")
    return float(value)


def _point(value, name: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} requires two normalized coordinates.")
    return (_finite_number(value[0], 0.0, 1.0, f"{name} x"),
            _finite_number(value[1], 0.0, 1.0, f"{name} y"))


@dataclass(frozen=True)
class SpotRemoval:
    """One circular heal/clone patch anchored to original-oriented [0, 1] coordinates."""

    id: str
    target: tuple[float, float]
    source: tuple[float, float]
    radius: float = DEFAULT_SPOT_RADIUS
    feather: float = DEFAULT_SPOT_FEATHER
    mode: str = "heal"
    opacity: float = DEFAULT_SPOT_OPACITY
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", self.id):
            raise ValueError("Invalid spot removal ID.")
        object.__setattr__(self, "target", _point(self.target, "Spot target"))
        object.__setattr__(self, "source", _point(self.source, "Spot source"))
        object.__setattr__(self, "radius", _finite_number(self.radius, MIN_SPOT_RADIUS, MAX_SPOT_RADIUS, "Spot size"))
        object.__setattr__(self, "feather", _finite_number(self.feather, 0.0, 1.0, "Spot feather"))
        if not isinstance(self.mode, str) or self.mode not in SPOT_MODES:
            raise ValueError("Spot mode must be 'heal' or 'clone'.")
        object.__setattr__(self, "opacity", _finite_number(self.opacity, 0.0, 1.0, "Spot opacity"))
        if type(self.enabled) is not bool:
            raise ValueError("Spot enabled flag must be boolean.")

    @property
    def active(self) -> bool:
        return self.enabled and self.opacity > 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "target": [self.target[0], self.target[1]],
            "source": [self.source[0], self.source[1]],
            "radius": self.radius,
            "feather": self.feather,
            "mode": self.mode,
            "opacity": self.opacity,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, value: dict) -> SpotRemoval:
        required = {"id", "target", "source", "radius", "feather", "mode", "opacity", "enabled"}
        if (not isinstance(value, dict) or set(value) != required
                or not isinstance(value["target"], list) or not isinstance(value["source"], list)):
            raise ValueError("Invalid spot removal parameters.")
        return cls(**value)


def validate_spots(items) -> tuple[SpotRemoval, ...]:
    if not isinstance(items, tuple) or len(items) > MAX_SPOTS or any(not isinstance(s, SpotRemoval) for s in items):
        raise ValueError(f"Use up to {MAX_SPOTS} immutable spot removals.")
    if len({s.id for s in items}) != len(items):
        raise ValueError("Spot removal IDs must be unique.")
    return items


def suggest_spot_source(target: tuple[float, float], radius: float, source_size: tuple[int, int] = (1, 1)) -> tuple[float, float]:
    """Choose a deterministic nearby source center inside safe image bounds."""
    tx, ty = _point(target, "Spot target")
    r = _finite_number(radius, MIN_SPOT_RADIUS, MAX_SPOT_RADIUS, "Spot size")
    w = max(1, int(source_size[0]))
    h = max(1, int(source_size[1]))
    ref = float(min(w, h))
    offset_px = max(2.6 * r * ref, 0.03 * ref)
    dx = offset_px / w
    dy = offset_px / h
    mx = min(0.25, (r * ref + 1.0) / w)
    my = min(0.25, (r * ref + 1.0) / h)
    # Prefer horizontal offsets first so vertical sky/horizon tonal gradients match cleanly.
    candidates = (
        (tx + dx, ty),
        (tx - dx, ty),
        (tx, ty + dy),
        (tx, ty - dy),
        (tx + dx * 0.7071, ty + dy * 0.7071),
        (tx - dx * 0.7071, ty + dy * 0.7071),
        (tx + dx * 0.7071, ty - dy * 0.7071),
        (tx - dx * 0.7071, ty - dy * 0.7071),
    )
    for cx, cy in candidates:
        if mx <= cx <= 1.0 - mx and my <= cy <= 1.0 - my:
            return (float(cx), float(cy))
    # Edge fallback: step toward image center while staying cleanly distinct from target.
    vx = 0.5 - tx
    vy = 0.5 - ty
    dist = math.hypot(vx * w, vy * h)
    if dist < 1e-6:
        return (min(1.0, tx + min(0.1, dx)), ty)
    step = min(dist, offset_px) / dist
    sx = max(0.0, min(1.0, tx + vx * step))
    sy = max(0.0, min(1.0, ty + vy * step))
    if math.hypot(sx - tx, sy - ty) < 1e-6:
        sx = 0.0 if tx >= 0.5 else 1.0
    return (float(sx), float(sy))


@dataclass(frozen=True)
class _PreparedSpot:
    spot: SpotRemoval
    tx_px: float
    ty_px: float
    sx_px: float
    sy_px: float
    shift_x: float
    shift_y: float
    r_px: float
    r_px2: float
    feather: float
    opacity: float
    mode: str
    gain_a: np.ndarray | None
    gain_bx: np.ndarray | None
    gain_by: np.ndarray | None


def _bilinear_sample(source_linear: np.ndarray, x_px: np.ndarray, y_px: np.ndarray,
                     source_size: tuple[int, int]) -> np.ndarray:
    """Sample source_linear at original-pixel coordinates using composition's pixel-center convention."""
    lin_h, lin_w = source_linear.shape[:2]
    src_w, src_h = source_size
    lx = np.clip(np.asarray(x_px, dtype=np.float64) * (lin_w / src_w) - 0.5, 0.0, max(0.0, lin_w - 1.0))
    ly = np.clip(np.asarray(y_px, dtype=np.float64) * (lin_h / src_h) - 0.5, 0.0, max(0.0, lin_h - 1.0))
    x0 = np.floor(lx).astype(np.intp)
    y0 = np.floor(ly).astype(np.intp)
    fx = (lx - x0).astype(np.float32)
    fy = (ly - y0).astype(np.float32)
    x1 = np.minimum(x0 + 1, lin_w - 1)
    y1 = np.minimum(y0 + 1, lin_h - 1)
    ofx = np.float32(1.0) - fx
    ofy = np.float32(1.0) - fy
    c00 = np.asarray(source_linear[y0, x0], dtype=np.float32)
    c10 = np.asarray(source_linear[y0, x1], dtype=np.float32)
    c01 = np.asarray(source_linear[y1, x0], dtype=np.float32)
    c11 = np.asarray(source_linear[y1, x1], dtype=np.float32)
    top = c00 * ofx[..., None] + c10 * fx[..., None]
    bot = c01 * ofx[..., None] + c11 * fx[..., None]
    return top * ofy[..., None] + bot * fy[..., None]


def _fit_heal_log_plane(spot: SpotRemoval, tx_px: float, ty_px: float, sx_px: float, sy_px: float,
                        r_px: float, source_linear: np.ndarray,
                        source_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a smooth log-linear RGB plane matching source perimeter tone/gradient to target perimeter."""
    src_w, src_h = source_size
    r1 = max(r_px + 1.0, 1.06 * r_px)
    r2 = max(r_px + 2.0, 1.16 * r_px)
    rho = 0.5 * (r1 + r2) / max(1e-6, r_px)

    t_x1 = tx_px + r1 * _COS_THETA
    t_y1 = ty_px + r1 * _SIN_THETA
    t_x2 = tx_px + r2 * _COS_THETA
    t_y2 = ty_px + r2 * _SIN_THETA
    s_x1 = sx_px + r1 * _COS_THETA
    s_y1 = sy_px + r1 * _SIN_THETA
    s_x2 = sx_px + r2 * _COS_THETA
    s_y2 = sy_px + r2 * _SIN_THETA

    t_ring = 0.5 * (_bilinear_sample(source_linear, t_x1, t_y1, source_size)
                    + _bilinear_sample(source_linear, t_x2, t_y2, source_size))
    s_ring = 0.5 * (_bilinear_sample(source_linear, s_x1, s_y1, source_size)
                    + _bilinear_sample(source_linear, s_x2, s_y2, source_size))

    delta = (np.log(np.maximum(t_ring, np.float32(0.0)) + _HEAL_EPS)
             - np.log(np.maximum(s_ring, np.float32(0.0)) + _HEAL_EPS))

    valid = ((t_x1 >= 0.0) & (t_x1 <= src_w) & (t_y1 >= 0.0) & (t_y1 <= src_h)
             & (s_x1 >= 0.0) & (s_x1 <= src_w) & (s_y1 >= 0.0) & (s_y1 <= src_h))
    if not np.any(valid):
        valid = np.ones(_RING_ANGLES, dtype=bool)

    med = np.median(delta[valid], axis=0)
    mad = np.median(np.abs(delta[valid] - med), axis=0)
    tol = np.maximum(np.float32(0.18), np.float32(3.0) * mad)
    inliers = valid & np.all(np.abs(delta - med) <= tol, axis=1)
    delta = np.clip(delta, med - np.float32(0.35), med + np.float32(0.35))
    delta = np.clip(delta, np.float32(-2.0), np.float32(2.0))

    n_inliers = int(np.count_nonzero(inliers))
    if n_inliers == _RING_ANGLES:
        a = delta.mean(axis=0)
        centered = delta - a
        bx = (2.0 / (_RING_ANGLES * rho)) * np.sum(centered * _COS_THETA[:, None], axis=0)
        by = (2.0 / (_RING_ANGLES * rho)) * np.sum(centered * _SIN_THETA[:, None], axis=0)
    elif n_inliers >= 8:
        ux = rho * _COS_THETA[inliers]
        uy = rho * _SIN_THETA[inliers]
        design = np.column_stack((np.ones(n_inliers, dtype=np.float64), ux, uy))
        # Regularize slope (bx, by) gently so partial-ring fits never extrapolate wildly.
        reg = np.diag([1e-4, 0.05, 0.05])
        ata = design.T @ design + reg
        aty = design.T @ delta[inliers].astype(np.float64)
        coef = np.linalg.solve(ata, aty)
        a, bx, by = coef[0], coef[1], coef[2]
    else:
        a = med
        bx = np.zeros(3, dtype=np.float32)
        by = np.zeros(3, dtype=np.float32)

    a = np.clip(np.asarray(a, dtype=np.float32), np.float32(-2.0), np.float32(2.0))
    bx = np.clip(np.asarray(bx, dtype=np.float32), np.float32(-0.8), np.float32(0.8))
    by = np.clip(np.asarray(by, dtype=np.float32), np.float32(-0.8), np.float32(0.8))
    return a, bx, by


def prepare_spots(spots: tuple[SpotRemoval, ...], source_linear: np.ndarray,
                  source_size: tuple[int, int]) -> tuple[_PreparedSpot, ...]:
    """Precompute strip-invariant spot coefficients strictly from source_linear."""
    src_w, src_h = source_size
    ref = float(min(src_w, src_h))
    prepared: list[_PreparedSpot] = []
    for spot in spots:
        if not spot.active:
            continue
        tx_px = spot.target[0] * src_w
        ty_px = spot.target[1] * src_h
        sx_px = spot.source[0] * src_w
        sy_px = spot.source[1] * src_h
        r_px = max(1e-6, spot.radius * ref)
        r_px2 = r_px * r_px
        if spot.mode == "heal":
            gain_a, gain_bx, gain_by = _fit_heal_log_plane(
                spot, tx_px, ty_px, sx_px, sy_px, r_px, source_linear, source_size,
            )
        else:
            gain_a = gain_bx = gain_by = None
        prepared.append(_PreparedSpot(
            spot=spot,
            tx_px=tx_px,
            ty_px=ty_px,
            sx_px=sx_px,
            sy_px=sy_px,
            shift_x=sx_px - tx_px,
            shift_y=sy_px - ty_px,
            r_px=r_px,
            r_px2=r_px2,
            feather=spot.feather,
            opacity=spot.opacity,
            mode=spot.mode,
            gain_a=gain_a,
            gain_bx=gain_bx,
            gain_by=gain_by,
        ))
    return tuple(prepared)


def _apply_spots_strip(strip: np.ndarray, prepared: tuple[_PreparedSpot, ...],
                       sx: np.ndarray, sy: np.ndarray, source_linear: np.ndarray,
                       source_size: tuple[int, int]) -> np.ndarray:
    sx_arr = np.asarray(sx, dtype=np.float64)
    sy_arr = np.asarray(sy, dtype=np.float64)
    sx_min = float(sx_arr.min())
    sx_max = float(sx_arr.max())
    sy_min = float(sy_arr.min())
    sy_max = float(sy_arr.max())
    out: np.ndarray | None = None

    for item in prepared:
        pad = item.r_px + 1e-3
        if (sx_max < item.tx_px - pad or sx_min > item.tx_px + pad
                or sy_max < item.ty_px - pad or sy_min > item.ty_px + pad):
            continue
        box = ((sx_arr >= item.tx_px - pad) & (sx_arr <= item.tx_px + pad)
               & (sy_arr >= item.ty_px - pad) & (sy_arr <= item.ty_px + pad))
        rows_any = np.any(box, axis=1)
        if not rows_any.any():
            continue
        cols_any = np.any(box, axis=0)
        r_lo = int(np.argmax(rows_any))
        r_hi = int(len(rows_any) - np.argmax(rows_any[::-1]))
        c_lo = int(np.argmax(cols_any))
        c_hi = int(len(cols_any) - np.argmax(cols_any[::-1]))

        sub_sx = sx_arr[r_lo:r_hi, c_lo:c_hi]
        sub_sy = sy_arr[r_lo:r_hi, c_lo:c_hi]
        dx = sub_sx - item.tx_px
        dy = sub_sy - item.ty_px
        d2 = dx * dx + dy * dy
        inside = (d2 < item.r_px2) if item.feather > 0.0 else (d2 <= item.r_px2)
        if not inside.any():
            continue

        if out is None:
            out = np.array(strip, dtype=np.float32, copy=True)

        sx_in = sub_sx[inside]
        sy_in = sub_sy[inside]
        dx_in = dx[inside]
        dy_in = dy[inside]

        if item.feather > 0.0:
            r_norm = np.sqrt(d2[inside]) / item.r_px
            t = np.clip((r_norm - (1.0 - item.feather)) / item.feather, 0.0, 1.0).astype(np.float32)
            w = np.float32(1.0) - t * t * (np.float32(3.0) - np.float32(2.0) * t)
            alpha = (w * np.float32(item.opacity))[..., None]
        else:
            alpha = np.full((len(sx_in), 1), np.float32(item.opacity), dtype=np.float32)

        src_rgb = _bilinear_sample(source_linear, sx_in + item.shift_x, sy_in + item.shift_y, source_size)
        if item.mode == "heal":
            assert item.gain_a is not None and item.gain_bx is not None and item.gain_by is not None
            ux = (dx_in / item.r_px).astype(np.float32)[..., None]
            uy = (dy_in / item.r_px).astype(np.float32)[..., None]
            log_gain = np.clip(item.gain_a + item.gain_bx * ux + item.gain_by * uy, np.float32(-2.5), np.float32(2.5))
            patched = np.maximum(np.float32(0.0), (src_rgb + _HEAL_EPS) * np.exp(log_gain) - _HEAL_EPS)
        else:
            patched = src_rgb

        sub_out = out[r_lo:r_hi, c_lo:c_hi]
        cur = sub_out[inside]
        sub_out[inside] = cur * (np.float32(1.0) - alpha) + patched * alpha

    return strip if out is None else out


def apply_spots(linear_rgb: np.ndarray, spots: tuple[SpotRemoval, ...], geometry: RenderGeometry,
                source_linear: np.ndarray, *, grid=None, prepared: tuple[_PreparedSpot, ...] | None = None,
                cancel=None) -> np.ndarray:
    """Apply active circular spots in linear RGB; zero-allocation short-circuit when idle."""
    if not spots or not any(s.active for s in spots):
        return linear_rgb
    if geometry.target_size != (linear_rgb.shape[1], linear_rgb.shape[0]):
        raise ValueError("Spot removal requires an explicit matching source sampling geometry.")
    if prepared is None:
        prepared = prepare_spots(spots, source_linear, geometry.source_size)
    if not prepared:
        return linear_rgb
    if grid is not None or linear_rgb.shape[0] <= 128:
        sx, sy = geometry.source_grid() if grid is None else grid
        return _apply_spots_strip(linear_rgb, prepared, sx, sy, source_linear, geometry.source_size)

    result: np.ndarray | None = None
    for top in range(0, linear_rgb.shape[0], 128):
        if cancel is not None and cancel.is_set():
            raise CancelledError()
        bottom = min(linear_rgb.shape[0], top + 128)
        rows = geometry.rows(top, bottom)
        sx, sy = rows.source_grid()
        strip_in = linear_rgb[top:bottom]
        strip_out = _apply_spots_strip(strip_in, prepared, sx, sy, source_linear, geometry.source_size)
        if strip_out is not strip_in:
            if result is None:
                result = np.array(linear_rgb, dtype=np.float32, copy=True)
            result[top:bottom] = strip_out
    return linear_rgb if result is None else result
