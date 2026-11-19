"""Task 3 v9.1 C03 Moirai 2.0, run by models_fm.moirai() in .venv-moirai (tickets/T03). Standalone: imports nothing
from this repo.

    .venv-moirai/bin/python -I moirai_side.py <ctx.npy> <out.npy> <weights dir>

ctx.npy: (rows, 512) contexts, left-padded with NaN. out.npy: the one-step 0.5 quantile per row. Moirai 2.0 outputs
fixed quantiles, not samples, so the forecast is its median (as Chronos-Bolt's).
"""

import sys

import numpy as np
import torch


def main(src, dst, weights, batch=64):
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    torch.set_num_threads(1)
    torch.manual_seed(340)
    module = Moirai2Module.from_pretrained(weights)
    model = Moirai2Forecast(module=module, prediction_length=1, context_length=512, target_dim=1,
                            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    q = list(module.quantile_levels).index(0.5)
    a = np.load(src)
    ctx = [r[np.isfinite(r)] for r in a]
    out = []
    for i in range(0, len(ctx), batch):
        p = model.predict(past_target=ctx[i:i + batch])          # (batch, quantiles, 1)
        out.append(p[:, q, 0])
    np.save(dst, np.concatenate(out))


if __name__ == "__main__":
    main(*sys.argv[1:4])
