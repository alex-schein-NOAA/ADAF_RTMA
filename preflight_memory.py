"""GPU memory probe: peak activation memory for a real forward+backward at full grid.

The pyramid's flagged risk (HANDOFF_multiscale.md §3) is memory: training is
activation-memory-bound and batch_size 2 is already the per-H100 ceiling for the flat
model. This measures peak allocation for both models at bs=1 and bs=2 under the real
training precision (bf16 autocast + channels_last), so batch_size/accum_steps is a
measured decision instead of a guess. No data needed -- random input of the right shape.

    python preflight_memory.py --configs ./config/params_default.yaml ./config/params_multiscale.yaml
"""

import argparse
import gc
import time

import torch

from models.encdec import build_model
from utils.YParams import YParams


def probe(cfg, batch_size, channels_last=True, amp_dtype=torch.bfloat16):
    params = YParams(cfg)
    h, w = params.img_size_y, params.img_size_x

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model(params).cuda()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n_params = sum(p.numel() for p in model.parameters())
    weights_mb = torch.cuda.max_memory_allocated() / 2**20

    x = torch.randn(batch_size, params.in_chans, h, w, device="cuda")
    tgt = torch.randn(batch_size, params.out_chans, h, w, device="cuda")
    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    def step():
        with torch.autocast("cuda", dtype=amp_dtype):
            loss = torch.nn.functional.mse_loss(model(x), tgt)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    try:
        for _ in range(2):  # 2nd step includes optimizer state -> the real steady-state peak
            step()
        peak = torch.cuda.max_memory_allocated() / 2**30

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_iter = 5
        for _ in range(n_iter):
            step()
        torch.cuda.synchronize()
        per_step = (time.perf_counter() - t0) / n_iter
        status = (f"peak {peak:6.1f} GiB   {per_step*1e3:7.0f} ms/step   "
                  f"{per_step/batch_size*1e3:7.0f} ms/sample")
    except torch.cuda.OutOfMemoryError:
        status = "OOM"

    del model, opt, x, tgt
    gc.collect()
    torch.cuda.empty_cache()
    return n_params, weights_mb, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["./config/params_default.yaml",
                             "./config/params_multiscale.yaml"])
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device: {torch.cuda.get_device_name(0)}  ({total:.0f} GiB)\n")

    for cfg in args.configs:
        for bs in args.batch_sizes:
            n, wmb, status = probe(cfg, bs)
            print(f"{cfg:38s} bs={bs}  params {n/1e6:6.2f} M "
                  f"weights+opt {wmb:7.1f} MiB   {status}")


if __name__ == "__main__":
    main()
