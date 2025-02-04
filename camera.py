
"""
The MIT License (MIT)
Copyright (c) 2021 NVIDIA CORPORATION
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
import time
import numpy as np
from threading import Thread

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GObject, Gst

Gst.init(None)


def _sanitize(element) -> Gst.Element:
    """
    Passthrough function which sure element is not `None`
    Returns `Gst.Element` or raises Error
    """
    if element is None:
        raise Exception("Element is none!")
    else:
        return element


def _make_element_safe(el_type: str, el_name=None) -> Gst.Element:
    """
    Creates a gstremer element using el_type factory.
    Returns Gst.Element or throws an error if we fail.
    This is to avoid `None` elements in our pipeline
    """

    # name=None parameter asks Gstreamer to uniquely name the elements for us
    el = Gst.ElementFactory.make(el_type, name=el_name)

    if el is not None:
        return el
    else:
        print(f"Pipeline element is None!")
        raise NameError(f"Could not create element {el_type}")


class Camera:
    def __init__(self, sensor_id, fps=None, shape_in=None, shape_out=None) -> None:

        # self._mainloop = GObject.MainLoop() # TODO: use GLib.MainLoop
        if any([fps, shape_in, shape_out]):
            self._pipeline = self._make_pipeline_with_resize(
                sensor_id, fps, shape_in, shape_out
            )
        else:
            self._pipeline = self._make_pipeline(sensor_id)
        self._pipeline.set_state(Gst.State.PLAYING)
        self.wait_ready()

    def stop(self):
        self._pipeline.set_state(Gst.State.NULL)

    def _make_pipeline_with_resize(
        self, sensor_id, fps=None, shape_in=None, shape_out=None
    ):

        pipeline = _sanitize(Gst.Pipeline())

        # Camera
        camera = _make_element_safe("nvarguscamerasrc")
        camera.set_property("sensor-id", sensor_id)

        # Input CF
        camera_cf = self._make_input_capsfilter(fps, shape_in)

        # nvvidconv
        conv = _make_element_safe("nvvidconv")

        # Output CF
        appsink_cf = self._make_output_capsfilter(shape_out)

        # Appsink
        self._appsink = appsink = _make_element_safe("appsink")

        # Add everything
        for el in [camera, camera_cf, conv, appsink_cf, appsink]:
            pipeline.add(el)

        camera.link(camera_cf)
        camera_cf.link(conv)
        conv.link(appsink_cf)
        appsink_cf.link(appsink)

        return pipeline

    def _make_pipeline(self, sensor_id):

        pipeline = _sanitize(Gst.Pipeline())

        cam = _make_element_safe("nvarguscamerasrc")
        cam.set_property("sensor-id", sensor_id)

        conv = _make_element_safe("nvvidconv")

        cf = _make_element_safe("capsfilter")
        cf.set_property(
            "caps", Gst.Caps.from_string("video/x-raw, format=(string)RGBA")
        )

        self._appsink = appsink = _make_element_safe("appsink")

        for el in [cam, conv, cf, appsink]:
            pipeline.add(el)

        cam.link(conv)
        conv.link(cf)
        cf.link(appsink)

        return pipeline

    @staticmethod
    def _make_input_capsfilter(fps, shape_in):

        caps_str = "video/x-raw(memory:NVMM), format=(string)NV12"

        if shape_in:
            W_in, H_in = shape_in
            caps_str += f", width=(int){W_in}, height=(int){H_in}"
        if fps:
            caps_str += f" framerate=(fraction){fps}/1"

        caps = Gst.Caps.from_string(caps_str)
        in_cf = _make_element_safe("capsfilter")
        in_cf.set_property("caps", caps)

        return in_cf

    @staticmethod
    def _make_output_capsfilter(shape_out):
        print(shape_out)
        if shape_out:
            W_out, H_out = shape_out
            caps = Gst.Caps.from_string(
                f"video/x-raw, width={W_out}, height={H_out}, format=(string)BGRx"
            )
        else:
            caps = Gst.Caps.from_string("video/x-raw, format=(string)RGBA")

        cf = _make_element_safe("capsfilter")
        cf.set_property("caps", caps)
        return cf

    def read(self):
        """
        Returns np.array or None
        """
        sample = self._appsink.emit("pull-sample")
        if sample is None:
            return None
        buf = sample.get_buffer()
        caps_format = sample.get_caps().get_structure(0)
        W, H = caps_format.get_value("width"), caps_format.get_value("height")
        C = 4  # Earlier we converted to RGBA
        buf2 = buf.extract_dup(0, buf.get_size())
        arr = np.ndarray(shape=(H, W, C), buffer=buf2, dtype=np.uint8)
        arr = arr[:, :, :3]  # RGBA -> RGB
        return arr

    def running(self):
        _, state, _ = self._pipeline.get_state(1)
        return True if state == Gst.State.PLAYING else False

    def wait_ready(self):
        while not self.running():
            time.sleep(0.1)


class CameraThread(Thread):
    def __init__(self, sensor_id, **kwargs) -> None:

        super().__init__()
        self._camera = Camera(sensor_id, **kwargs)
        self._should_run = True
        self._image = self._camera.read()
        self.start()

    def run(self):
        while self._should_run:
            self._image = self._camera.read()

    @property
    def image(self):
        # NOTE: if we care about atomicity of reads, we can add a lock here
        return self._image

    def stop(self):
        # TODO: this should be threading.Event
        self._should_run = False
        self._camera.stop()


if __name__ == "__main__":

    camera = Camera(0, shape_in=(1920, 1080), shape_out=(224, 224))

    for _ in range(10):
        start = time.perf_counter()
        arr = camera.read()
        print(
            f"Latency: {time.perf_counter() - start} Image shape: {arr.shape} Image mean: {arr.mean()}"
        )

    camera.stop()
