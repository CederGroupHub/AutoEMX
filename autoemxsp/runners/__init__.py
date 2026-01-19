#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package initializer for autoemxsp.runners

Automatically imports all submodules and re-exports their public symbols.
Symbols must be declared in the module's __all__ to be exported.
"""

from pathlib import Path
import pkgutil
import importlib

__all__ = []

# Directory containing this package
_package_dir = Path(__file__).parent

# Automatically discover and import submodules
for _, module_name, _ in pkgutil.iter_modules([str(_package_dir)]):
    module = importlib.import_module(f"{__name__}.{module_name}")

    # Re-export only explicitly declared public symbols
    if hasattr(module, "__all__"):
        for name in module.__all__:
            obj = getattr(module, name)
            globals()[name] = obj
            __all__.append(name)