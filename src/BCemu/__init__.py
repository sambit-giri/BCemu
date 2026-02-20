'''
Emulator is a Python package for constructing emulators.

You can also get documentation for all routines directory from
the interpreter using Python's built-in help() function.
For example:
>>> import BCemu
>>> help(BCemu.use_emul)
'''
import sys
import importlib.metadata

try:
    __version__ = importlib.metadata.version("BCemu")
except importlib.metadata.PackageNotFoundError:
    # This happens if the user just cloned the repo and imported the folder
    # without running `pip install .` or `pip install -e .`
    __version__ = "unknown"
    
from .BaryonEffectsEmulator import *
from . import download
# from . import kpls 
from .datasets import *
import smt


#Suppress warnings from zero-divisions and nans
import numpy
numpy.seterr(all='ignore')
