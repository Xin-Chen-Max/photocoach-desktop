"""Immutable manual gradients and explicit source-coordinate sampling contexts."""
from dataclasses import dataclass, replace
import math
import re
import numpy as np
from .composition import Composition, canvas_size, crop_bounds, output_size


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

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", self.id):
            raise ValueError("Invalid gradient ID.")
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 40 or any(ord(c)<32 for c in self.name):
            raise ValueError("Gradient names require 1–40 printable characters.")
        object.__setattr__(self, "name", self.name.strip())
        for key in ("start", "end"):
            point = getattr(self, key)
            if not isinstance(point, (tuple, list)) or len(point) != 2:
                raise ValueError("A gradient endpoint needs two coordinates.")
            object.__setattr__(self, key, tuple(number(v,0,1) for v in point))
        if math.dist(self.start, self.end) < 1e-5:
            raise ValueError("Drag farther to create a gradient transition.")
        if type(self.enabled) is not bool or type(self.inverted) is not bool:
            raise ValueError("Gradient switches must be boolean.")
        for key, limit in (("exposure_ev",3),("highlights",1),("shadows",1)):
            object.__setattr__(self,key,number(getattr(self,key),-limit,limit))

    @property
    def active(self):
        return self.enabled and bool(self.exposure_ev or self.highlights or self.shadows)

    def to_dict(self):
        return dict(id=self.id,name=self.name,start=list(self.start),end=list(self.end),
                    enabled=self.enabled,inverted=self.inverted,exposure_ev=self.exposure_ev,
                    highlights=self.highlights,shadows=self.shadows)

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or set(value) != {"id","name","start","end","enabled","inverted","exposure_ev","highlights","shadows"}:
            raise ValueError("Invalid gradient parameters.")
        return cls(**value)


@dataclass(frozen=True)
class RadialGradient(LinearGradient):
    """An ellipse whose opposite bounding corners use original-image coordinates."""
    feather: float = .5

    def __post_init__(self):
        super().__post_init__()
        if any(abs(a-b)<1e-5 for a,b in zip(self.start,self.end)):
            raise ValueError("Drag wider and taller to create a radial mask.")
        object.__setattr__(self,'feather',number(self.feather,0,1))

    def to_dict(self):
        return dict(super().to_dict(),kind='radial',feather=self.feather)

    @classmethod
    def from_dict(cls,value):
        required={'id','name','start','end','enabled','inverted','exposure_ev','highlights','shadows','kind','feather'}
        if not isinstance(value,dict) or set(value)!=required or value['kind']!='radial':
            raise ValueError('Invalid radial mask parameters.')
        return cls(**{k:v for k,v in value.items() if k!='kind'})


def gradient_from_dict(value):
    if isinstance(value,dict) and value.get('kind')=='radial':
        return RadialGradient.from_dict(value)
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
    def full(cls,source_size,composition,target_size):
        return cls(tuple(source_size),composition,(0,0,*output_size(source_size,composition)),tuple(target_size))

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
        return (c*x[None,:]+s*y[:,None]+self.source_size[0]/2,
                -s*x[None,:]+c*y[:,None]+self.source_size[1]/2)


def source_point(point,source_size,composition,*,inverse=False):
    """Map a composed canvas pixel center to original pixels, or reverse it."""
    cw,ch=canvas_size(source_size,composition.angle)
    cl,ct,_,_=crop_bounds(source_size,composition)
    a=math.radians(composition.angle); c,s=math.cos(a),math.sin(a)
    if inverse:
        x,y=point[0]-source_size[0]/2,point[1]-source_size[1]/2
        return (c*x-s*y+cw/2-cl,s*x+c*y+ch/2-ct)
    x,y=point[0]+cl-cw/2,point[1]+ct-ch/2
    return (c*x+s*y+source_size[0]/2,-s*x+c*y+source_size[1]/2)


def gradient_weight(gradient,geometry,grid=None):
    sx,sy=geometry.source_grid() if grid is None else grid
    w,h=geometry.source_size
    if isinstance(gradient,RadialGradient):
        cx=(gradient.start[0]+gradient.end[0])*w/2
        cy=(gradient.start[1]+gradient.end[1])*h/2
        rx=abs(gradient.end[0]-gradient.start[0])*w/2
        ry=abs(gradient.end[1]-gradient.start[1])*h/2
        radius=np.sqrt(((sx-cx)/rx)**2+((sy-cy)/ry)**2)
        if gradient.feather:
            t=np.clip((radius-(1-gradient.feather))/gradient.feather,0,1).astype(np.float32)
            weight=np.float32(1)-t*t*(np.float32(3)-np.float32(2)*t)
        else:
            weight=(radius<=1).astype(np.float32)
        return np.float32(1)-weight if gradient.inverted else weight
    ax,ay=gradient.start[0]*w,gradient.start[1]*h
    dx,dy=(gradient.end[0]-gradient.start[0])*w,(gradient.end[1]-gradient.start[1])*h
    t=np.clip(((sx-ax)*dx+(sy-ay)*dy)/(dx*dx+dy*dy),0,1).astype(np.float32)
    t=t*t*(np.float32(3)-np.float32(2)*t)
    return t if gradient.inverted else np.float32(1)-t


def tone_fields(gradients,geometry,highlights,shadows):
    grid=geometry.source_grid()
    shape=(geometry.target_size[1],geometry.target_size[0])
    ev=np.zeros(shape,np.float32)
    hl=np.full(shape,highlights,np.float32); sh=np.full(shape,shadows,np.float32)
    for g in gradients:
        if g.active:
            weight=gradient_weight(g,geometry,grid)
            ev+=weight*np.float32(g.exposure_ev)
            hl+=weight*np.float32(g.highlights)
            sh+=weight*np.float32(g.shadows)
    np.clip(hl,-1,1,out=hl); np.clip(sh,-1,1,out=sh)
    return ev,hl,sh


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
    for top in range(0,h,128):
        bottom=min(h,top+128); block=result[top:bottom]
        if spec.mode=="clipping":
            image=pixels[top:bottom]
            block[np.any(image==255,axis=2)]=(255,70,70,170)
            block[np.all(image==0,axis=2)]=(65,115,255,170)
        elif spec.mode=="mask" and spec.gradient:
            weight=gradient_weight(spec.gradient,geometry.rows(top,bottom))
            block[...,:3]=(70,205,210)
            block[...,3]=np.rint(weight*90).astype(np.uint8)
    result.setflags(write=False)
    return result
