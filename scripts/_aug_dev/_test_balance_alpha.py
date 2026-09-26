"""S1 sampling: InContextPairs.balance_alpha.

Checks, without loading a corpus (a bare InContextPairs object carries only the fields _pick reads):
  1. alpha 0 -> draw_cdf is None and _pick reproduces the ORIGINAL uniform-over-rigs draw, same RNG consumption
     (byte-identical batches for every existing arm);
  2. alpha 0.5 / 1.0 -> empirical rig frequencies follow (target count)^alpha within tolerance, every entry reachable,
     multiplicity honoured;
  3. refusal: alpha < 0 (the balance-off refusal lives in __init__, which a bare object bypasses; not covered here).
Plus the corpus-level expectation on the UniMate common cut (manifest only), the numbers quoted in the config.

usage: python scripts/_aug_dev/_test_balance_alpha.py
"""
from __future__ import annotations
import collections, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data.incontext_pairs import InContextPairs, draw_cdf_for  # noqa: E402


def bare(draw_types, by_type, alpha, balance=True):
    ds = InContextPairs.__new__(InContextPairs)
    ds.balance = balance
    ds.draw_types = list(draw_types)
    ds.by_type = by_type
    ds.draw_cdf = draw_cdf_for(ds.draw_types, ds.by_type, alpha) if balance else None
    ds.index = [(t, i) for t in sorted(by_type) for i in by_type[t]["targets"]]
    return ds


def original_pick(ds, rng):
    ot = ds.draw_types[int(rng.integers(len(ds.draw_types)))]
    tg = ds.by_type[ot]["targets"]
    return ot, int(tg[int(rng.integers(len(tg)))])


def main():
    by_type = {"lone_a": {"targets": [0]}, "lone_b": {"targets": [1]}, "four": {"targets": [2, 3, 4, 5]},
               "sixteen": {"targets": list(range(6, 22))}, "big": {"targets": list(range(22, 122))}}
    types = sorted(by_type)
    n_ok = 0
    # 1. alpha 0: identical draws, identical RNG stream
    ds0 = bare(types, by_type, 0.0)
    assert ds0.draw_cdf is None
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    a = [ds0._pick(i, r1) for i in range(5000)]
    b = [original_pick(ds0, r2) for i in range(5000)]
    assert a == b, "alpha 0 changed the draw"
    assert r1.random() == r2.random(), "alpha 0 changed RNG consumption"
    n_ok += 1
    # 2. alpha 0.5 / 1.0: frequencies ~ n^alpha, every entry reachable
    for alpha in (0.5, 1.0):
        ds = bare(types, by_type, alpha)
        rng = np.random.default_rng(11)
        cnt = collections.Counter(ds._pick(i, rng)[0] for i in range(200000))
        w = {t: len(by_type[t]["targets"]) ** alpha for t in types}; Z = sum(w.values())
        for t in types:
            exp = 200000 * w[t] / Z
            assert abs(cnt[t] - exp) < 4 * np.sqrt(exp) + 30, (alpha, t, cnt[t], exp)
        assert set(cnt) == set(types)
        # per-clip: under alpha 1 every clip is equally likely
        if alpha == 1.0:
            rng5 = np.random.default_rng(5)
            per = collections.Counter(ds._pick(i, rng5)[1] for i in range(200000))
            vals = np.array([per[i] for i in range(122)]); assert vals.min() > 0.6 * vals.mean() and vals.max() < 1.4 * vals.mean()
    n_ok += 1
    # multiplicity: a repeated entry is weighted once per repeat
    ds = bare(types + ["lone_a"], by_type, 0.5)
    rng = np.random.default_rng(3); cnt = collections.Counter(ds._pick(i, rng)[0] for i in range(100000))
    assert abs(cnt["lone_a"] / cnt["lone_b"] - 2.0) < 0.15, cnt
    # the last cell is pinned so u -> 1 cannot fall off the end
    cdf = draw_cdf_for(types, by_type, 0.5); assert cdf[-1] == 1.0 and len(cdf) == len(types)
    n_ok += 1
    # 3. refusals
    for bad in (-0.5,):
        try:
            draw_cdf_for(types, by_type, bad); raise AssertionError("negative alpha accepted")
        except ValueError:
            pass
    n_ok += 1
    # corpus expectation on the UniMate common cut (manifest only)
    split_p = REPO / "runs/_unimate/common_v2_split.json"
    if split_p.is_file():
        n = collections.Counter(o.split("-", 1)[0] for o in json.load(open(split_p))["train_official_ids"])
        D = 55984
        for alpha in (0.0, 0.5):
            w = {r: c ** alpha for r, c in n.items()}; Z = sum(w.values())
            lone = D * 1.0 / Z; big = max(n.values()); bigr = next(r for r in n if n[r] == big)
            print(f"[corpus] alpha={alpha}: lone clip {lone:.1f}x/epoch, {big}-clip rig's clip {D * w[bigr] / Z / big:.2f}x/epoch, "
                  f"lone-rig share {100 * sum(D * w[r] / Z for r in n if n[r] == 1) / D:.0f}%")
    print(f"ALL PASS ({n_ok} groups)")


if __name__ == "__main__":
    main()
