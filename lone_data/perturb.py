"""Degrade a frame the way the path to a live camera degrades it.

A checkpoint trained on one episode scores ~90% when replayed against the
byte-exact training tensors and collapses to two distinct actions on a live
camera. The gap is not latency -- the replay carries the same chunking and the
same queue -- it is that live pixels are never the pixels training saw:

  training / replay  camera -> resize -> AV1 encode -> [stored] -> AV1 decode
  live deployment    camera -> resize
  live, remote       camera -> resize -> JPEG q80 -> network -> JPEG decode

plus whatever the room's lighting has done since the episode was recorded. Each
of those is a fraction of a percent of pixel value. Measured against this
checkpoint, 0.3% was enough to move a prediction 42x further than the policy's
own sampling noise and flip the command from "drive the lower arm" to "stop".

So these are not decorations on the replay. Being able to dial them up while the
arm is running is the difference between knowing a policy is brittle and knowing
how brittle, in the units the deployment actually varies by.

Order is contrast, then brightness, then noise, then JPEG -- the order the real
path applies them: the sensor's response first, then transmission.
"""

import cv2
import numpy as np

# Contrast pivots here rather than on the frame's own mean, so the transform does
# not change with the scene and two runs stay comparable.
MID_GREY = 128.0


class FeedPerturbation:
    """Brightness, contrast, noise and JPEG applied to a uint8 RGB frame.

    Fields are plain attributes so a GUI can write them between frames; `apply`
    reads them each call and short-circuits when everything is at its identity.
    """

    def __init__(self, brightness=1.0, contrast=1.0, noise=0.0, jpeg_quality=0, seed=0):
        self.brightness = float(brightness)
        self.contrast = float(contrast)
        self.noise = float(noise)
        self.jpeg_quality = int(jpeg_quality)
        self._rng = np.random.default_rng(seed)

    @property
    def active(self):
        return (self.brightness != 1.0 or self.contrast != 1.0
                or self.noise > 0.0 or self.jpeg_quality > 0)

    def reset(self):
        self.brightness = self.contrast = 1.0
        self.noise = 0.0
        self.jpeg_quality = 0

    def apply(self, rgb):
        """uint8 HWC RGB in, uint8 HWC RGB out. Returns the input untouched when idle."""
        if not self.active:
            return rgb
        out = rgb.astype(np.float32)
        if self.contrast != 1.0:
            out = (out - MID_GREY) * self.contrast + MID_GREY
        if self.brightness != 1.0:
            out *= self.brightness
        if self.noise > 0.0:
            # Fresh noise per frame, not a fixed pattern: a static offset is
            # something a policy can be robust to by accident.
            out += self._rng.normal(0.0, self.noise, out.shape)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if self.jpeg_quality > 0:
            # Through BGR because that is the order cv2's codec expects, and the
            # round trip has to match what stream_camera.py actually does.
            bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                out = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return out

    def mean_abs_diff(self, rgb):
        """Mean absolute pixel change this perturbation makes, in 0-255 units.

        The readout that makes the sensitivity legible: it puts the input change
        on the same scale as the action change it produced.
        """
        if not self.active:
            return 0.0
        return float(np.abs(self.apply(rgb).astype(np.int16) - rgb.astype(np.int16)).mean())

    def describe(self):
        if not self.active:
            return "clean"
        bits = []
        if self.contrast != 1.0:
            bits.append(f"contrast x{self.contrast:.2f}")
        if self.brightness != 1.0:
            bits.append(f"brightness x{self.brightness:.2f}")
        if self.noise > 0.0:
            bits.append(f"noise s={self.noise:.1f}")
        if self.jpeg_quality > 0:
            bits.append(f"JPEG q{self.jpeg_quality}")
        return "  ·  ".join(bits)

    def as_dict(self):
        return {
            "brightness": round(self.brightness, 4),
            "contrast": round(self.contrast, 4),
            "noise": round(self.noise, 3),
            "jpeg_quality": self.jpeg_quality,
        }
