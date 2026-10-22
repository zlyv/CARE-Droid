"""Read-only timing of frozen Direct-C43 on existing transformed tensors (CPU)."""
from __future__ import annotations

import resource
import time

import numpy as np

from common import OUTPUT, original, read_csv, write_csv, write_text
from contrastive_training import inputs


def main() -> None:
    mc, v4, _, data = original()
    torch = v4.torch
    torch.set_num_threads(4)
    device = torch.device('cpu')
    routing = {(r['tag'], r['sha256']): int(r['routed']) for r in read_csv(OUTPUT / 'expert_capability.csv')}
    result = []
    for tag in ('T2_C1', 'T3_C1', 'T4_C1'):
        start = time.perf_counter()
        obj, args, _, valid, global_pair, bags, graphs, _ = inputs(mc, v4, data, tag, 900000)
        preprocessing_seconds = time.perf_counter() - start
        model = v4.RoleGraphConditionedEvidence(24, 'RGCondAttn', args).to(device)
        model.load_state_dict(obj['state_dict'])
        model.eval()
        graph = v4.graph_tensors(graphs[1], device)
        global_tensor = torch.as_tensor(global_pair[1], dtype=torch.float32, device=device)
        host_bags = [torch.as_tensor(b, dtype=torch.float32, device=device) for b in bags[1]]
        identifiers = [str(data['ids'][i]) for i in valid]
        route_indices = [i for i, sid in enumerate(identifiers) if routing[(tag, sid)]]
        if len(route_indices) not in (99, 100):
            raise RuntimeError(f'unexpected route count for {tag}: {len(route_indices)}')

        def predict(index_list):
            result_probs = []
            with torch.no_grad():
                for i in index_list:
                    one = {key: value[i] for key, value in graph.items()}
                    logit, _, _ = model.forward_with_trace(
                        global_tensor[i], torch.zeros((), device=device), one, host_bags[i]
                    )
                    result_probs.append(float(torch.sigmoid(logit)))
            return result_probs

        predict(route_indices[:3])
        for scope, indices in [('routed_top25', route_indices), ('all_validation', list(range(len(valid))))]:
            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                probabilities = predict(indices)
                times.append(time.perf_counter() - t0)
                if len(probabilities) != len(indices) or not np.isfinite(probabilities).all():
                    raise RuntimeError('invalid timed prediction')
            result.append(dict(
                tag=tag,
                scope=scope,
                n=len(indices),
                input_preprocessing_seconds=preprocessing_seconds,
                inference_seconds_median=float(np.median(times)),
                inference_seconds_min=min(times),
                inference_seconds_max=max(times),
                inference_milliseconds_per_APK=1000 * float(np.median(times)) / len(indices),
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                device='cpu_4_threads',
                original_model='T2-T4 seed900000 frozen M0',
                original_APK_parsing_included=False,
                StageI_and_router_included=False,
                background_training_included=False,
            ))
        print('ETBC_COST', tag, result[-2]['inference_seconds_median'], result[-1]['inference_seconds_median'], flush=True)
    write_csv(OUTPUT / 'cached_inference_cost.csv', result)
    write_text(OUTPUT / 'cached_inference_cost_boundary.md',
               '# 缓存张量推理成本边界\n\n'
               '该计时在同一 CPU 四线程环境下，对每个滚动窗口的冻结 M0（种子900000）分别重复三次，'
               '只包括模型前向传播；先独立计时复用缓存的预处理。'
               'Top25% 与全量验证 APK 的比较具有同设备、同模型、同缓存口径。'
               '它不是从原 APK 提取特征、重建 Background Bank、Stage-I 与 Router 的完整系统时延，'
               '也不是新 M1–M5 各模型的独立 GPU 推理时延。\n')


if __name__ == '__main__':
    main()
