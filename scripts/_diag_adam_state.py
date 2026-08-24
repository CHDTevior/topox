#!/usr/bin/env python3
"""Is the second moment collapsing?

Crash is locked to OPTIMIZER STEPS, not data or epochs: run7 saw 3x the data of run6 yet died at
g27000 vs g32000. Adam's update is m/(sqrt(v)+eps). If a parameter's gradient stays near zero for
a long time, v decays toward zero; a single rare large gradient then produces an update of order
m/sqrt(v), which can be enormous. v needs STEPS to decay -- exactly the observed dependence.
"""
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
opt = ck["opt"]; st = opt["state"]
print(f"{sys.argv[1]}  epoch={ck['epoch']} gstep={ck['gstep']}  states={len(st)}")
rows = []
for i, s in st.items():
    if "exp_avg_sq" not in s:
        continue
    v = s["exp_avg_sq"].double().flatten()
    m = s["exp_avg"].double().flatten()
    sv = v.sqrt()
    ratio = m.abs() / (sv + 1e-8)          # the actual Adam step magnitude, pre-lr
    rows.append((float(ratio.max()), float(v.min()), float(v.median()), float(sv.min()),
                 float(ratio.median()), int(i), s.get("step", None)))
rows.sort(reverse=True)
print(f"\n{'max m/(sqrt(v)+e)':>18} {'min v':>12} {'median v':>12} {'min sqrt(v)':>12} "
      f"{'median ratio':>13}  param#")
for r in rows[:12]:
    print(f"{r[0]:>18.4g} {r[1]:>12.4g} {r[2]:>12.4g} {r[3]:>12.4g} {r[4]:>13.4g}  {r[5]}")
allmax = max(r[0] for r in rows)
tiny = sum(1 for r in rows if r[3] < 1e-8)      # sqrt(v) below eps -> eps dominates
print(f"\nWORST Adam step (pre-lr): {allmax:.4g}")
print(f"tensors whose min sqrt(v) < eps=1e-8: {tiny}/{len(rows)}")
step = rows[0][6]
print(f"optimizer step count: {step}")
