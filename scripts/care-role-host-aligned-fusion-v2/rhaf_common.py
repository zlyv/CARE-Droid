#!/usr/bin/env python3
"""Uniquely named loader protecting this experiment from historical generic common-module imports."""
from pathlib import Path
_path=Path(__file__).with_name('common.py')
exec(compile(_path.read_text(encoding='utf-8'),str(_path),'exec'),globals(),globals())
