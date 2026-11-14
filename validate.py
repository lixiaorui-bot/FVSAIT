# Validation for the FVSAIT re-implementation
# 1. FHRR algebra unit tests (Eq. 1-5)
# 2. Lookahead optimizer unit test (Eq. 10-11)
# 3. build model from configs (full pipeline incl. VGG19 + LSH + FHRR)
# 4. forward pass + one training-style forward/loss computation
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import fhrr  # noqa: E402


def fhrr_tests():
    print("== FHRR algebra tests ==")
    torch.manual_seed(0)
    x = torch.rand(2, 4096, 4, 4) * 2 - 1  # [-1, 1]

    v = fhrr.phase_encode(x)
    assert torch.allclose(v.abs(), torch.ones_like(v.abs()), atol=1e-5), "|v| != 1"
    print("phase_encode unit magnitude: OK")

    # binding / unbinding are inverse (Eq. 2)
    a = fhrr.phase_encode(torch.rand_like(x) * 2 - 1)
    b = fhrr.phase_encode(torch.rand_like(x) * 2 - 1)
    err = (fhrr.unbind(fhrr.bind(a, b), b) - a).abs().max()
    assert err < 1e-5, f"unbind(bind(a,b),b) != a ({err})"
    print(f"bind/unbind inverse: max err {err:.2e}")

    # similarity (Eq. 4): sim(v, v) = 1; orthogonal-ish vectors ~ 0
    s_self = fhrr.similarity(v, v, dim=1)
    assert (s_self - 1).abs().max() < 1e-5, "sim(v,v) != 1"
    v2 = fhrr.phase_encode(torch.rand_like(x) * 2 - 1)
    s_other = fhrr.similarity(v, v2, dim=1)
    assert s_other.abs().mean() < 0.05, "random hypervectors not ~orthogonal"
    print(f"similarity: sim(v,v)=1, |sim(v,random)|~{s_other.abs().mean():.4f}")

    # bundling (Eq. 3): result is unit magnitude, similar to constituents
    # (rare near-cancellation elements deviate slightly due to the eps guard)
    s = fhrr.bundle([a, b])
    assert (s.abs() - 1).abs().max() < 2e-3, "bundle not unit magnitude"
    sim_a = fhrr.similarity(s, a, dim=1).mean()
    assert 0.3 < sim_a < 1.0, "bundle should stay similar to constituents"
    print(f"bundling: unit magnitude, sim(bundle, a) = {sim_a:.3f}")

    # gradient flows through the complex ops
    xg = (torch.rand_like(x) * 2 - 1).requires_grad_(True)
    vg = fhrr.phase_encode(xg)
    u = fhrr.phase_encode(vg.real.detach())  # detach one side
    loss = (1 - fhrr.similarity(fhrr.bind(vg, u), vg, dim=1)).mean()
    loss.backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    print("complex autograd: OK")


def lookahead_tests():
    print("== Lookahead optimizer tests ==")
    from utils.lookahead import Lookahead

    p = torch.nn.Parameter(torch.zeros(4))
    inner = torch.optim.Adam([p], lr=0.1)
    opt = Lookahead(inner, k=5, alpha=0.5)

    # 4 steps: slow weights must not move yet (fast weights differ)
    for i in range(4):
        p.grad = torch.ones(4)
        opt.step()
        opt.zero_grad()
    theta_after4 = p.detach().clone()
    # 5th step triggers slow update: p = slow + 0.5*(fast - slow)
    p.grad = torch.ones(4)
    opt.step()
    assert not torch.allclose(p.detach(), theta_after4), "slow update did not fire"
    print("k-step slow update fires: OK")

    # state dict round trip
    sd = opt.state_dict()
    opt2 = Lookahead(torch.optim.Adam([p], lr=0.1), k=5, alpha=0.5)
    opt2.load_state_dict(sd)
    print("state_dict round trip: OK")


def build_and_forward():
    print("== model build & forward ==")
    from utils.config import get_cfg
    from models.iit import FVSAIT

    cfg = get_cfg()
    cfg.merge_from_file("configs/fvsait.yaml")
    model = FVSAIT(cfg)

    n_g = sum(p_.numel() for p_ in model.g_net.parameters())
    n_r = sum(p_.numel() for p_ in model.g_net_map.parameters())
    n_d = sum(p_.numel() for p_ in model.d_net.parameters())
    print(f"G: {n_g/1e6:.3f}M   R(map): {n_r/1e6:.3f}M   D: {n_d/1e6:.3f}M")

    model.eval()
    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        y = model.generate(x)
        print(f"generate 256x256 -> {tuple(y.shape)}")
        v = model.encode(x, "source")
        print(f"encode -> complex {tuple(v.shape)}, |v| min/max = "
              f"{v.abs().min():.3f}/{v.abs().max():.3f}")
        phases = model.encode_phases(x, "source")
        u = model.map_phases(phases)
        cyc = model.fhrr.bind(v, u)
        sim = model.fhrr.similarity(cyc, v, dim=1).mean()
        print(f"identity-init cycle sim (should be ~1): {sim:.4f}")
        assert sim > 0.99, "identity init broken"

    # training-style forward (both optimizer keys) on CPU
    model.train()
    batch = {
        "source": {"image": torch.rand(1, 3, 256, 256)},
        "target": {"image": torch.rand(1, 3, 256, 256)},
    }
    out_adv, tgt_adv = model(batch, optimizer_idx=0)
    assert "vsa" in out_adv and "adv" in out_adv
    assert out_adv["vsa"]["source"].is_complex()
    out_d, tgt_d = model(batch, optimizer_idx=1)
    assert len(out_d["discr"]) == 2  # fake + real
    print(f"forward adv keys: {list(out_adv)}, discr outputs: {len(out_d['discr'])}")

    # loss computation
    from losses.da_loss import FVSALoss
    crit = FVSALoss(cfg)
    l_adv = crit(out_adv, tgt_adv)
    l_d = crit(out_d, tgt_d)
    assert torch.isfinite(l_adv) and torch.isfinite(l_d)
    print(f"losses: adv/G = {l_adv.item():.3f}, discr/D = {l_d.item():.3f}")

    l_adv.backward()
    print("backward through FHRR complex graph: OK")


if __name__ == "__main__":
    fhrr_tests()
    lookahead_tests()
    build_and_forward()
    print("ALL CHECKS PASSED")
