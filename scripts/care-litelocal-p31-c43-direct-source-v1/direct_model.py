#!/usr/bin/env python3
from __future__ import annotations

import torch


def make_direct_class(base_class):
    """Use the unchanged C43 scalar head as a direct classification logit.

    The base implementation computes raw_head, then
    base_logit + max_delta*tanh(raw_head). A forward hook captures raw_head so
    this arm can return it directly without duplicating C43 feature assembly.
    """
    class DirectC43(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not hasattr(self, "delta_head"):
                raise RuntimeError("BLOCKED_BY_CONTRACT: C43 scalar head missing")
            self._direct_raw_logit = None

            def capture(_module, _inputs, output):
                self._direct_raw_logit = output.reshape(())
                return None

            self.delta_head.register_forward_hook(capture)

        def forward(self, *args, **kwargs):
            _bounded_logit, diagnostic = super().forward(*args, **kwargs)
            if self._direct_raw_logit is None:
                raise RuntimeError("BLOCKED_BY_CONTRACT: raw C43 head was not captured")
            raw_logit = self._direct_raw_logit
            # Preserve the diagnostic width expected by v4, while making the
            # first field explicitly the direct raw classification logit.
            direct_diagnostic = torch.cat([
                raw_logit.reshape(1), diagnostic.reshape(-1)[1:]
            ])
            return raw_logit, direct_diagnostic

    DirectC43.__name__ = "DirectC43RelationQuery"
    return DirectC43
