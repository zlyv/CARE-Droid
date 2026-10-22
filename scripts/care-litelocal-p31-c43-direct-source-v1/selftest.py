#!/usr/bin/env python3
from __future__ import annotations

import torch
from torch import nn

from direct_model import make_direct_class


class ToyBounded(nn.Module):
    def __init__(self):
        super().__init__()
        self.delta_head = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.delta_head.weight.copy_(torch.tensor([[1.0, -1.0]]))

    def forward(self, values, base_logit):
        raw = self.delta_head(values).reshape(())
        final = base_logit.reshape(()) + 2.25 * torch.tanh(raw)
        return final, torch.stack([2.25 * torch.tanh(raw), raw])


def main():
    Direct = make_direct_class(ToyBounded)
    model = Direct()
    values = torch.tensor([2.0, .5])
    output, diagnostic = model(values, torch.tensor(-8.0))
    assert torch.allclose(output, torch.tensor(1.5))
    assert torch.allclose(diagnostic[0], torch.tensor(1.5))
    output.backward()
    assert model.delta_head.weight.grad is not None
    print("P31_C43_DIRECT_SOURCE_SELFTEST_PASS")
    print("bounded_residual_formula_used=false direct_raw_logit=true gradient_preserved=true")


if __name__ == "__main__":
    main()
