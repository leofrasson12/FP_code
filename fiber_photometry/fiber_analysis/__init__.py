"""
fiber_analysis
==============
Top‐level package for fiber photometry analysis.
"""

__version__ = "0.1.0"

# Expose submodules
from .io import *
from .preprocess import *
from .peth import *
from .denoise import *
from .detect import *
from .plots import *
