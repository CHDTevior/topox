#!/usr/bin/env python3
"""Quantitative jitter attribution for the pilot renders (user 2026-08-28: persistent micro-
jitter survives sigma_min 0.05 + MSE + oriented logit-normal AND 32 ODE steps -- analyze).

For each rendered clip: sample the model (cfg_text 2, steps configurable), recover world
positions through BOTH decodings, and decompose against GT:
  [A] temporal spectrum per joint (detrended rFFT @30fps), band energies 0-2/2-5/5-10/10-15 Hz
      -> WHERE in frequency the excess lives, gen/GT per band;
  [B] per-joint acceleration ratio -> WHICH joints jitter (with chain depth from parents);
  [C] RIC(direct) vs FK recovery high-band energy on the SAME gen -> rotation-representation
      amplification hypothesis;
  [D] GT's own high-band share -> source-data jitter baseline;
  [E] normalization std of the jitteriest joints' position channels -> small-denominator
      amplification hypothesis;
  [F] acceleration p95/p50 -> spiky vs uniform jitter.
Read-only; writes a text report. Usage:
  python scripts/_analyze_jitter.py --ckpt <pt> --steps 32 --out <report.txt>
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ktjd17_incontext import Ktjd17Base, ktjd17_split_names          # noqa: E402
from src.data.incontext_pairs import InContextPairs, collate                   # noqa: E402
from src.data.anytop_dataset import _STD_FLOOR                                 # noqa: E402
from src.models.v2.dit_motion import InContextMotionDiT, sample                # noqa: E402
from scripts.v2_render_incontext import world_of_ktjd                          # noqa: E402
from torch.utils.data import DataLoader

RIGS = ["PZ_Bairds_Tapir_Male", "PZ_California_Sea_Lion_Male", "PZ_Dall_Sheep_Male",
        "PZ_African_Elephant_Male", "PZ_Saltwater_Crocodile_Male", "PZ_Arctic_Wolf_Male"]
BANDS = [(0, 2), (2, 5), (5, 10), (10, 15)]
FPS = 30.0

def band_energy(w, fps=FPS):
    """w [T,J,3] -> per-band energy [len(BANDS), J]: detrended rFFT power summed per band."""
    T = w.shape[0]
    tgrid = np.arange(T)[:, None, None]
    # remove per-joint linear trend so travel does not read as 0Hz energy
    p1 = np.polyfit(np.arange(T), w.reshape(T, -1), 1)
    trend = (p1[0][None] * tgrid.reshape(T, 1) + p1[1][None]).reshape(w.shape)
    x = w - trend
    F = np.fft.rfft(x, axis=0)
    P = (np.abs(F) ** 2).sum(-1)                       # [T//2+1, J]
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    out = np.zeros((len(BANDS), w.shape[1]))
    for i, (lo, hi) in enumerate(BANDS):
        m = (freqs >= lo) & (freqs < hi)
        out[i] = P[m].sum(0)
    return out

def acc_mag(w):
    a = w[2:] - 2 * w[1:-1] + w[:-2]
    return np.linalg.norm(a, axis=-1)                  # [T-2, J]

def depth_of(parents):
    d = np.zeros(len(parents), dtype=int)
    for j, p in enumerate(parents):
        d[j] = 0 if p < 0 else d[p] + 1
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--cfg_text", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ktjd_root", default="dataset/ktjd17_pzh312_noik_v2")
    a = ap.parse_args()
    dev = "cuda:0"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    ca = ck["args"]; ca = vars(ca) if not isinstance(ca, dict) else ca
    m = InContextMotionDiT(in_ch=17, dim=ca["dim"], depth=ca["depth"], n_heads=ca["heads"],
                           d_text=4096, d_joint_sem=4096,
                           use_struct_feats=bool(ca.get("struct_feats", False)),
                           use_dir_bias=bool(ca.get("dir_bias", False)),
                           qk_norm=bool(ca.get("qk_norm", False))).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    base = Ktjd17Base(a.ktjd_root, caption_emb_cache=ca["caption_cache"],
                      joint_semantics=ca["joint_sem"], percell_stats=ca["ktjd_percell_stats"],
                      exclude_clips=ca.get("exclude_clips") or None, texts_json=ca["texts_json"])
    names = ktjd17_split_names(a.ktjd_root, exclude=ca.get("exclude_clips") or None)
    ds = InContextPairs(base, names["train"], names["train"], balance_skeletons=False, seed=7,
                        emit_fk_fields=True, emit_graph_v2=True, demo_rest=True, demo_frames=1,
                        target_frames=240)
    rep = [f"# jitter attribution | ckpt={a.ckpt} ep={ck['epoch']} steps={a.steps} "
           f"cfg={a.cfg_text} fps={FPS}"]
    agg = {"band_ratio": [], "fk_vs_ric": [], "gt_high_share": []}
    for rig in RIGS:
        pos = [i for i, (ot, _) in enumerate(ds.index) if ot == rig]
        if not pos:
            rep.append(f"\n== {rig}: NOT FOUND =="); continue
        # longest clip, mirroring --pick longest
        i_best = max(pos, key=lambda i: min(int(base[ds.index[i][1]]["num_frames"]), ds.Tt))
        item = ds[i_best]
        b = collate([item])
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        torch.manual_seed(7)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=bool(ca.get("bf16", False))):
            gen = sample(m, b["x"][..., :17], b["is_target"], a.steps, cfg_text=a.cfg_text,
                         demo_frames=1, joint_bias=b["joint_bias"],
                         frame_valid=b["frame_valid"], joint_valid=b["joint_valid"],
                         text=b["text"], joint_sem=b["joint_sem"],
                         struct_feats=b.get("struct_feats"), updown=b.get("updown"))
        T = min(int(base[ds.index[i_best][1]]["num_frames"]), ds.Tt)
        J = int(b["n_joints"][0])
        g18 = gen[0, 1:1 + T - 1, :J].float().cpu().numpy()      # drop demo frame
        t18 = b["x"][0, 1:1 + T - 1, :J].float().cpu().numpy()
        gen_ric, gen_fk = world_of_ktjd(g18, base, rig, strict_gt=False)
        gt_ric, gt_fk = world_of_ktjd(t18, base, rig, strict_gt=True)
        sk = base._skeleton(rig)
        dep = depth_of(sk["parents"])
        # [A] band energies (FK path, the deployed one)
        bg, bt = band_energy(gen_fk), band_energy(gt_fk)
        tot_g, tot_t = bg.sum(1), bt.sum(1)
        ratios = tot_g / np.maximum(tot_t, 1e-12)
        agg["band_ratio"].append(ratios)
        # [D] GT's own high-band share
        gt_high = float(tot_t[2:].sum() / max(tot_t.sum(), 1e-12))
        agg["gt_high_share"].append(gt_high)
        # [B] jitteriest joints by acceleration ratio
        ag, at = acc_mag(gen_fk).mean(0), acc_mag(gt_fk).mean(0)
        jr = ag / np.maximum(at, 1e-9)
        top = np.argsort(-jr)[:5]
        # [C] FK vs RIC high-band on the same gen
        hg_fk = band_energy(gen_fk)[2:].sum()
        hg_ric = band_energy(gen_ric)[2:].sum()
        fkr = float(hg_fk / max(hg_ric, 1e-12)); agg["fk_vs_ric"].append(fkr)
        # [E] normalization std of the top joints' position channels (0:3)
        _, sd_r = base._pc[rig]
        sd_pos = sd_r[:J, 0:3].mean(-1)
        # [F] spikiness
        af = acc_mag(gen_fk); spik = float(np.percentile(af, 95) / max(np.percentile(af, 50), 1e-9))
        gspik = acc_mag(gt_fk); gsp = float(np.percentile(gspik, 95) / max(np.percentile(gspik, 50), 1e-9))
        rep.append(f"\n== {rig} (T={T} J={J}) ==")
        rep.append("  [A] gen/GT band-energy ratio: " + " ".join(
            f"{lo}-{hi}Hz={r:.2f}" for (lo, hi), r in zip(BANDS, ratios)))
        rep.append(f"  [C] gen FK/RIC high-band(>5Hz) energy ratio: {fkr:.2f}")
        rep.append(f"  [D] GT own high-band(>5Hz) share of energy: {gt_high:.4f}")
        rep.append(f"  [F] acc p95/p50 gen={spik:.1f} GT={gsp:.1f}")
        rep.append("  [B] top-5 jitter joints (acc gen/GT | chain-depth | pos-std):")
        for j in top:
            rep.append(f"      j{j:3d} ratio={jr[j]:7.2f} depth={dep[j]:2d} "
                       f"gtacc={at[j]:.5f} std={sd_pos[j]:.4f}")
    br = np.stack(agg["band_ratio"]).mean(0)
    rep.append("\n== AGGREGATE ==")
    rep.append("  mean gen/GT band ratios: " + " ".join(
        f"{lo}-{hi}Hz={r:.2f}" for (lo, hi), r in zip(BANDS, br)))
    rep.append(f"  mean gen FK/RIC high-band ratio: {np.mean(agg['fk_vs_ric']):.2f}")
    rep.append(f"  mean GT high-band share: {np.mean(agg['gt_high_share']):.4f}")
    Path(a.out).write_text("\n".join(rep) + "\n")
    print("\n".join(rep))

if __name__ == "__main__":
    main()
