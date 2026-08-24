#!/usr/bin/env python3
"""Localise the phase transition per tensor.

Comparing a checkpoint from just BEFORE the blow-up against one from just AFTER tells us which
part of the network destabilises first. The same method localised run3's failure in minutes.
"""
import sys, torch
from collections import defaultdict
A, B = sys.argv[1], sys.argv[2]
a = torch.load(A, map_location="cpu", weights_only=False)["model"]
b = torch.load(B, map_location="cpu", weights_only=False)["model"]
rows = []
for k in a:
    if k not in b or not a[k].is_floating_point():
        continue
    x, y = a[k].double(), b[k].double()
    na = x.norm().item()
    rel = (y - x).norm().item() / (na + 1e-12)
    rows.append((rel, k, na, y.norm().item(), float(y.abs().max())))
rows.sort(reverse=True)
print(f"{A.split('/')[-1]} -> {B.split('/')[-1]}")
print(f"{'rel change':>11}  {'|w_before|':>11}  {'|w_after|':>11}  {'max|w_after|':>12}  tensor")
for rel, k, na, nb, mx in rows[:18]:
    print(f"{rel:>11.4f}  {na:>11.3f}  {nb:>11.3f}  {mx:>12.4g}  {k}")
mod = defaultdict(lambda: [0.0, 0.0, 0])
for rel, k, na, nb, mx in rows:
    m = k.split(".")[0] if not k.startswith("blocks") else "blocks"
    mod[m][0] += (nb - na) ** 2; mod[m][1] = max(mod[m][1], rel); mod[m][2] += 1
print("\nby module:  worst-rel   n_tensors")
for m, (_, wr, n) in sorted(mod.items(), key=lambda kv: -kv[1][1]):
    print(f"  {m:<22} {wr:>9.4f}   {n}")
nonfinite = [k for k in b if b[k].is_floating_point() and not torch.isfinite(b[k]).all()]
print(f"\nnon-finite tensors in AFTER: {len(nonfinite)}  {nonfinite[:5]}")
