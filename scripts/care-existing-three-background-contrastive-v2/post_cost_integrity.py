"""One more protected-asset check after the optional cached inference timing."""
from __future__ import annotations

from pathlib import Path

from common import OUTPUT, asset, read_json, write_json


def main() -> None:
    before = read_json(OUTPUT / 'baseline_asset_manifest_before.json')
    changed = []
    for row in before:
        path = Path(row['path'])
        if not path.is_file():
            changed.append(dict(path=str(path), reason='missing'))
            continue
        after = asset(path, row['purpose'])
        if any(after[key] != row[key] for key in ('size', 'mtime_ns', 'sha256')):
            changed.append(dict(path=str(path), before=row, after=after))
    status = 'PASS' if not changed else 'FAIL'
    write_json(OUTPUT / 'baseline_integrity_check_post_cost.json', dict(
        status=status, asset_count=len(before), changed=changed,
        scope='listed reused assets only; checked after all model and optional cost runs',
    ))
    print('ETBC_POST_COST_INTEGRITY', status, len(before), flush=True)
    if changed:
        raise RuntimeError('protected asset changed')


if __name__ == '__main__':
    main()
