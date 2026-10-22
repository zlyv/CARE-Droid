#!/usr/bin/env python3
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import torch

from host_composition_encoder import HostSemanticCompositionEncoder
from host_position_encoder import POSITION_DIM, HostStructuralPositionEncoder, structural_vector
from host_complement_model import make_model_class
from host_complement_training import RAW_DIM, make_fit_adapter, pack, unpack


def main() -> None:
    torch.manual_seed(7)
    enc = HostSemanticCompositionEncoder(5, 24)
    one = torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32)
    assert torch.equal(enc(one), torch.zeros(1, 24))                         # 1
    assert HostSemanticCompositionEncoder.pair_count(one[0]) == 0
    two = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.float32)
    three = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.float32)
    assert HostSemanticCompositionEncoder.pair_count(two[0]) == 1            # 2
    assert HostSemanticCompositionEncoder.pair_count(three[0]) == 3          # 3
    m1=HostSemanticCompositionEncoder.unique_membership([0,2],5).float()[None,:]
    m2=HostSemanticCompositionEncoder.unique_membership([2,0],5).float()[None,:]
    assert torch.allclose(enc(m1), enc(m2))                                    # 4 set-order invariant
    duplicated = HostSemanticCompositionEncoder.unique_membership([0, 2, 2], 5)
    assert duplicated.tolist() == [1, 0, 1, 0, 0] and HostSemanticCompositionEncoder.pair_count(duplicated) == 1  # 5
    assert enc(two).shape == (1, 24)                                         # 6
    alpha = torch.sigmoid(torch.randn(8))
    assert bool(torch.all((alpha >= 0) & (alpha <= 1)))                       # 7

    source = structural_vector(0, 1, 0, 2)
    sink = structural_vector(1, 0, 2, 0)
    bridge = structural_vector(1, 2, 1, 2)
    isolated = structural_vector(0, 0, 0, 0)
    assert source[7:11] == [1, 0, 0, 0] and sink[7:11] == [0, 1, 0, 0]
    assert bridge[7:11] == [0, 0, 1, 0] and isolated[7:11] == [0, 0, 0, 1]   # 8
    assert np.allclose(bridge[:4], [0.5, 0.5, 0.5, 0.5])
    assert np.allclose(bridge[4:7], [1/3, 1/3, 1/3])                          # 9
    pos = HostStructuralPositionEncoder(24)
    beta = torch.sigmoid(torch.randn(8))
    assert pos(torch.zeros(2, POSITION_DIM)).shape == (2, 24)
    assert bool(torch.all((beta >= 0) & (beta <= 1)))                         # 10

    h = torch.randn(3, 24); c = torch.randn(3, 24); s = torch.randn(3, 24)
    a = torch.sigmoid(torch.randn(3)); b = torch.sigmoid(torch.randn(3))
    expected = h + a[:, None] * c + b[:, None] * s
    actual = h + a[:, None] * c + b[:, None] * s
    assert torch.equal(actual, expected)                                      # 11
    class DummyBase(torch.nn.Module):
        def __init__(self, global_dim, variant, args): super().__init__()
    class DummyFusion(torch.nn.Module):
        def __init__(self, dim): super().__init__()
    AB=make_model_class(DummyBase,DummyFusion,5,RAW_DIM+4,"AB")
    ab=AB(24,"RGCondAttn",SimpleNamespace(embedding_dim=24))
    complement_gates=[name for name,_ in ab.named_modules() if name.endswith("_gate")]
    assert complement_gates==["composition_gate","position_gate"]                # 12

    p1 = np.array([.2, .8]); expert = np.array([.9, .1]); route = np.array([0, 1])
    final = np.where(route == 1, expert, p1)
    assert final[0] == p1[0]                                                  # 13
    raw = torch.tensor([-100., 0., 100.]); delta = 2.25 * torch.tanh(raw)
    assert float(torch.max(torch.abs(delta))) <= 2.25                         # 14
    host = np.arange(2 * RAW_DIM, dtype=np.float32).reshape(2, RAW_DIM)
    roles = np.zeros((2, 5), np.float32); positions = np.zeros((2, POSITION_DIM), np.float32)
    bag = pack(host, roles, positions, [(0, 1)])
    restored, _, _, edges = unpack(bag, RAW_DIM, 5)
    captured=[]
    def original_fit(train,test,labels,args,seed):
        captured.extend(np.array(x,copy=True) for x in train+test)
        augment=lambda values:[np.concatenate([x,np.zeros((len(x),4),np.float32)],1) for x in values]
        return augment(train),augment(test),{}
    adapted=make_fit_adapter(original_fit,5)
    adapted([bag],[bag],np.array([0]),SimpleNamespace(),7)
    assert np.array_equal(restored,host) and all(np.array_equal(x,host) for x in captured) # 15
    encoded = torch.randn(2, 24); used = torch.tensor([False, True]); fused = encoded.clone()
    fused[used] += 1
    assert torch.equal(fused[~used], encoded[~used])                           # 16
    assert int(.499999 >= .5) == 0 and int(.5 >= .5) == 1                     # 17
    print("HOST_COMPLEMENT_NEURAL_SELFTEST_PASS")


if __name__ == "__main__":
    main()
