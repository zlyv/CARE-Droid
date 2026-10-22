"""Sample GPU use during the independent experiment; never inspect APK content."""
from __future__ import annotations

import csv
import subprocess
import time

from common import OUTPUT


def main() -> None:
    output = OUTPUT / 'resource_samples.csv'
    log = OUTPUT / 'full_pipeline.log'
    if output.exists():
        raise RuntimeError(f'resource sample already exists: {output}')
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch_seconds', 'gpu_index', 'memory_used_mib', 'memory_total_mib', 'gpu_utilization_pct'])
        writer.writeheader()
        for _ in range(720):
            query = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                text=True, capture_output=True, check=True, timeout=10,
            )
            now = time.time()
            for line in query.stdout.splitlines():
                fields = [part.strip() for part in line.split(',')]
                if len(fields) != 4:
                    raise RuntimeError(f'unexpected nvidia-smi row: {line!r}')
                writer.writerow(dict(epoch_seconds=now, gpu_index=int(fields[0]), memory_used_mib=int(fields[1]), memory_total_mib=int(fields[2]), gpu_utilization_pct=int(fields[3])))
            stream.flush()
            if log.exists() and 'ETBC_FULL_DONE' in log.read_text(encoding='utf-8')[-4096:]:
                break
            time.sleep(5)


if __name__ == '__main__':
    main()
