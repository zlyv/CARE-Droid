#!/usr/bin/env python3
"""Synthetic unit tests for reachability, loops, same-block order and failures."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'care-local-control-constraint-feasibility-v2'
sys.path.insert(0, str(PARENT))
from control_core import build_cfg
from witness_core import relation, strongly_connected_components, witness_pair


def row(offset, name='nop', length=2, **extra):
    value = dict(offset=offset, name=name, length=length, output='', registers=[], payload=False)
    value.update(extra)
    return value


def main():
    straight = build_cfg([row(0), row(2), row(4, 'return-void')])
    assert relation(straight, 0, 2)[0] == 'A_TO_B'
    pair = witness_pair(straight, 0, 2)
    assert pair['same_basic_block'] is True
    assert witness_pair(straight, 2, 0)['relation'] == 'B_TO_A'

    branch = build_cfg([row(0, 'if-eqz', target=4, registers=[0]), row(2), row(4, 'return-void')])
    assert relation(branch, 2, 4)[0] == 'A_TO_B'

    loop = build_cfg([row(0), row(2, 'if-eqz', target=0, registers=[0]), row(4, 'return-void')])
    components, nodes = strongly_connected_components(loop)
    assert 0 in nodes and 2 in nodes
    assert relation(loop, 0, 2)[0] == 'BIDIRECTIONAL'

    empty = build_cfg([])
    assert empty['status'] == 'UNKNOWN'
    assert relation(empty, 0, 2)[0] == 'UNKNOWN'
    bad = build_cfg([row(0, 'if-eqz', target=99, registers=[0])])
    assert bad['status'] == 'UNKNOWN'
    print('CFG_ROLEGRAPH_WITNESS_SELFTEST_PASS')


if __name__ == '__main__': main()

