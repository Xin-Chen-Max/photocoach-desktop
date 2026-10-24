"""Bounded final-quality strip acceleration preserves the existing contract."""
import threading
import time
import numpy as np
import pytest

from photocoach import preview
from photocoach.color_mixer import ColorMixer
from photocoach.composition import Composition, output_size
from photocoach.state import EditValues


@pytest.mark.parametrize("step", [1, 3])
def test_existing_hsl_pixels_exact_in_both_execution_modes(step):
    source=np.random.default_rng(43).random((1300,1900,3),dtype=np.float32)
    source.setflags(write=False)
    values=EditValues(composition=Composition(angle=3.25),shadows=.3,highlights=-.4,
        color_mixer=ColorMixer().with_value(2,2,15).with_value(5,1,-10))
    size=output_size((1900,1300),values.composition)
    request=preview.DetailRequest(0,size,(0,0,*size),step)
    serial=preview.render_region(source,values,request,workers=1)
    parallel=preview.render_region(source,values,request,workers=4)
    np.testing.assert_array_equal(serial,parallel)


def test_partial_submit_failure_joins_started_strips(monkeypatch):
    if preview._STRIP_WORKERS < 2: pytest.skip('Requires two CPU workers')
    started,finished=threading.Event(),threading.Event()
    def render(source,*args,**kwargs):
        started.set();time.sleep(.03);finished.set()
        return source.copy()
    monkeypatch.setattr(preview,'render_edit',render)
    original=preview._STRIP_POOL.submit
    calls=0
    def submit(*args):
        nonlocal calls
        calls+=1
        if calls==2:
            assert started.wait(2)
            raise RuntimeError('simulated submission failure')
        return original(*args)
    monkeypatch.setattr(preview._STRIP_POOL,'submit',submit)
    image=np.zeros((1300,1900,3),np.float32)
    with pytest.raises(RuntimeError,match='submission failure'):
        preview.render_region(image,EditValues(),preview.DetailRequest(0,(1900,1300),(0,0,1900,1300)))
    assert finished.is_set()
