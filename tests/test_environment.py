"""Checks the environment from the README's Getting Started is actually installed.

No hardware, dataset or checkpoint is touched.
"""

import importlib
import importlib.metadata as metadata
import shutil
import subprocess
import sys

import pytest

DEPENDENCIES = [
    "numpy", "cv2", "PIL", "scipy", "sklearn", "pygame", "serial", "matplotlib",
    "wandb", "tqdm", "tkinter",   # tkinter: every GUI here is Tk
    "transformers",               # lerobot[pi]
    "av",                         # lerobot[dataset]
    "accelerate",                 # lerobot[training]
    "lone_data",                  # this repo, installed with pip install -e
]


def test_python_is_at_least_3_12():
    assert sys.version_info >= (3, 12), sys.version


@pytest.mark.parametrize("module", DEPENDENCIES)
def test_dependency_imports(module):
    importlib.import_module(module)


def test_pinned_versions():
    import torch

    assert metadata.version("lerobot") == "0.6.1"
    assert metadata.version("torchcodec") == "0.3.0"  # must match torch 2.7
    assert torch.__version__.startswith("2.7")

    from torchcodec.decoders import VideoDecoder  # noqa: F401  (loads the native extension)


def test_ffmpeg_is_installed():
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, 'conda install -c conda-forge "ffmpeg=7"'
    subprocess.run([ffmpeg, "-version"], capture_output=True, check=True)


def test_policies_are_constructible():
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    assert ACTConfig().type == "act"
    assert PI05Config().type == "pi05"
