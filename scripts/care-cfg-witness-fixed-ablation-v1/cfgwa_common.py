#!/usr/bin/env python3
"""Uniquely named loader for this experiment's common contract.

Historical CARE-Droid modules intentionally reuse the generic name ``common``.
Loading this file under ``cfgwa_common`` prevents those imports from replacing
the experiment's already frozen helpers.
"""
from pathlib import Path

_source = Path(__file__).with_name('common.py')
exec(compile(_source.read_text(encoding='utf-8'), str(_source), 'exec'), globals(), globals())
