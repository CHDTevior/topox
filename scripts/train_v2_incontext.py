#!/usr/bin/env python3
"""Train the in-context [demo | target] motion DiT on TrueBones (run-1 baseline, single GPU).

Everything here was fixed by the preflight/smoke chain, not by preference:
  data     frozen protocol splits (canonical-topology level); [demo 64 | target 240] = T 304;
           target slot covers the longest clip (237) so captions always describe the trained frames
  loss     CFM x0-prediction; grouped objective KIMODO_GAMMAS x sqrt(N_i/N_total) (root undiluted,
           gradient share independent of group size); `valid` mandatory
  sampling skeleton-balanced (Trex 72 clips must not outweigh Chicken 2)
  run-1    NO classifier-free guidance -- this run is the baseline the CFG run compares against

Validation = bucket A (seen rigs, unseen clips), with a RESET RNG stream each pass so every val
sees the identical demo/crop choices and the numbers are comparable across epochs.
Checkpoints are written atomically (tmp + rename); best is tracked on val flow loss.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.anytop_dataset import AnyTopDataset                               # noqa: E402
from src.data.incontext_pairs import (InContextPairs, collate, read_split,      # noqa: E402
                                      truebones_types, pzh_types, DEMO_FRAMES, TARGET_FRAMES)
from src.models.v2.dit_motion import (InContextMotionDiT, cfm_loss,             # noqa: E402
                                      KIMODO_GAMMAS)


def to_dev(b, dev):
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}


def cond_of(b):
    return dict(joint_bias=b["joint_bias"], frame_valid=b["frame_valid"],
                joint_valid=b["joint_valid"], text=b["text"], joint_sem=b["joint_sem"])


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)                     # atomic on POSIX: no torn checkpoint on kill


def reset_val_stream(ds):
    """Make the next val pass replay the exact demo/crop choices of every previous pass."""
    ds._wrng_key = None


class fixed_torch_rng:
    """Deterministic torch RNG scope for validation.

    Resetting the DATA stream is not enough: cfm_loss draws t ~ rand and x0 ~ randn from the
    GLOBAL torch RNG, so two val passes over identical batches would still differ. Inside this
    scope the noise/t draws are identical every pass; global state is restored on exit so
    training randomness is untouched.
    """
    def __init__(self, seed): self.seed = seed
    def __enter__(self):
        self.cpu = torch.get_rng_state()
        self.gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)
    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu)
        if self.gpu is not None:
            torch.cuda.set_rng_state_all(self.gpu)


def connectivity_probe(model, b, demo_frames=DEMO_FRAMES):
    """P5: |dLoss/d input| per conditioning path. Zero = dead branch (v1's undetected failure).
    Magnitudes are comparable only against earlier probes of THIS run."""
    was_training = model.training
    model.train()
    c = cond_of(b)
    for k in ("text", "joint_sem"):
        c[k] = c[k].detach().clone().requires_grad_(True)
    xin = b["x"].detach().clone().requires_grad_(True)
    loss = cfm_loss(model, xin, is_target=b["is_target"], valid=b["valid"],
                    gammas=KIMODO_GAMMAS, **c)
    g_x, g_t, g_s = torch.autograd.grad(loss, [xin, c["text"], c["joint_sem"]])
    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return {"demo": float(g_x[:, :demo_frames].norm()),
            "target": float(g_x[:, demo_frames:].norm()),
            "text": float(g_t.norm()), "joint_sem": float(g_s.norm())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/animo4d_L4TB_plus_human_v4b272neutral")
    ap.add_argument("--splits_dir", default="data/holdout_splits_v1")
    ap.add_argument("--joint_sem", default="data/joint_semantics_llm2vec_v1.npz")
    ap.add_argument("--caption_cache", default="data/anytop_caption_llm2vec_v4b272neutral_multi")
    ap.add_argument("--texts_json", default="motion_texts_by_file_clean_v1.json")
    ap.add_argument("--out", default="runs/v2_incontext_run1")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    # ---- run-3 scale knobs (all defaults preserve run-1 behaviour bit-for-bit) ----
    ap.add_argument("--corpus", choices=("truebones", "pzh"), default="truebones",
                    help="pzh = Planet-Zoo + HumanML3D, no TrueBones (312 rigs / 89.5k train clips)")
    ap.add_argument("--balance", choices=("rig", "clip"), default="rig",
                    help="rig = uniform-skeleton draws (run-1); clip = natural source proportions")
    ap.add_argument("--random_caption", action="store_true",
                    help="rotate ALL captions per clip (median 3) instead of cap0 only")
    ap.add_argument("--demo_frames", type=int, default=DEMO_FRAMES)
    ap.add_argument("--target_frames", type=int, default=TARGET_FRAMES)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=0,
                    help="linear lr warmup over N optimizer steps (big-model stability)")
    ap.add_argument("--val_max_batches", type=int, default=0,
                    help=">0: cap validation at N batches (fixed deterministic subset) so peer "
                         "ranks are not parked behind a long rank-0 val")
    ap.add_argument("--val_every_steps", type=int, default=0,
                    help=">0: validate/checkpoint every N steps INSTEAD of every val_every epochs")
    ap.add_argument("--ckpt_snapshot_steps", type=int, default=0,
                    help=">0: periodic epNNN-style snapshots every N steps instead of ckpt_every epochs")
    ap.add_argument("--t_sampler", choices=("uniform", "logitnormal"), default="uniform")
    ap.add_argument("--v_space", action="store_true", help="JiT velocity-space loss (clamped 1/(1-t)^2)")
    ap.add_argument("--p_drop_text", type=float, default=0.0)
    ap.add_argument("--p_drop_demo", type=float, default=0.0)
    ap.add_argument("--p_drop_both", type=float, default=0.0)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "run-1 is a GPU run; refusing to silently train on CPU"
    # ---- DDP is opt-in via torchrun's env; absent WORLD_SIZE keeps the single-GPU path
    # bit-identical (run-1 and its crash-resume must not change behaviour). Cross-alloc same-node
    # specifics (static rendezvous, NCCL_P2P/SHM disable, IB socket) live in the LAUNCHER, not here.
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    # A stray RANK/LOCAL_RANK without WORLD_SIZE must NOT leak into seeding or is_main:
    # single-GPU behaviour is pinned bit-identical to the pre-DDP trainer.
    rank = int(os.environ.get("RANK", "0")) if ddp else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if ddp else 0
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}" if ddp else "cuda"
    is_main = rank == 0
    torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank)
    out = Path(a.out)
    if is_main:
        out.mkdir(parents=True, exist_ok=True)
    if ddp:
        dist.barrier()

    # ---------------- data ----------------
    cond = pickle.load(open(f"{a.data_root}/_cond_normalized_J144.pkl", "rb"))
    types = truebones_types(cond.keys()) if a.corpus == "truebones" else pzh_types(cond.keys())
    names = {k: read_split(a.splits_dir, k) for k in ("train", "val")}
    base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=a.caption_cache,
                         random_caption=a.random_caption, augment=False, joint_semantics=a.joint_sem,
                         species_whitelist=types, splits_dir=a.splits_dir,
                         texts_json_name=a.texts_json)
    ds_tr = InContextPairs(base, names["train"], names["train"], object_types=types,
                           demo_frames=a.demo_frames, target_frames=a.target_frames,
                           balance_skeletons=(a.balance == "rig"), seed=a.seed)
    ds_va = InContextPairs(base, names["val"], names["train"], object_types=types,
                           demo_frames=a.demo_frames, target_frames=a.target_frames,
                           balance_skeletons=False, seed=a.seed + 1)
    print(f"[train] {len(ds_tr)} targets / {len(ds_tr.types)} rigs / {ds_tr.pair_count()} pairs "
          f"| bucket-A val {len(ds_va)} targets / {len(ds_va.types)} rigs", flush=True)

    # Under DDP a DistributedSampler partitions the INDEX SPACE (734 -> ~183/rank -> 22 steps at
    # B8, 704 global draws vs 728 single-GPU). Balanced _pick ignores the indices themselves; the
    # sampler only meters how many batches each rank runs, and set_epoch() is a harmless no-op kept
    # for convention.
    tr_sampler = DistributedSampler(ds_tr, shuffle=True, drop_last=True) if ddp else None
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=(tr_sampler is None),
                       sampler=tr_sampler, num_workers=a.num_workers,
                       collate_fn=collate, pin_memory=True,
                       # drop_last: an "epoch" is INTENTIONALLY full batches only (734 targets ->
                       # 91x8 = 728 draws single-GPU; 22x8x4 = 704 under 4-rank DDP). Balanced
                       # _pick ignores the index, so no target is systematically excluded.
                       drop_last=True,
                       persistent_workers=a.num_workers > 0)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, num_workers=0,
                       collate_fn=collate, pin_memory=True)

    # ---------------- model ----------------
    model = InContextMotionDiT(in_ch=13, dim=a.dim, depth=a.depth, n_heads=a.heads,
                               d_text=4096, d_joint_sem=4096).to(dev)
    raw_model = model
    if ddp:
        # find_unused_parameters: bp_mlp (the shelved blueprint pathway, param indices 17-20)
        # never enters the graph, and DDP's reducer otherwise waits forever for its gradients --
        # the H200 smoke caught exactly this. Costs a small per-step graph walk; excising bp_mlp
        # outright would break strict state_dict loading of every run-1 checkpoint.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    n_par = sum(p.numel() for p in raw_model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    start_ep, best_val, gstep = 0, float("inf"), 0
    if a.resume:
        # Resume is a STATISTICAL continuation, not bit-exact: DataLoader worker streams re-derive
        # from a fresh base_seed after restart. What must NOT drift silently is the config -- a
        # resumed run with different dims/lr/data would corrupt the ckpt lineage, so critical
        # fields are hard-checked. Torch/numpy RNG states are restored best-effort on top.
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)  # our own ckpt; contains numpy RNG state, rejected by 2.6's weights_only default
        old_args = ck.get("args", {})
        crit = ("dim", "depth", "heads", "batch", "lr", "seed", "data_root", "splits_dir",
                "joint_sem", "caption_cache", "texts_json",
                # run-3 trajectory-defining knobs: silently flipping any of these mid-run would
                # change the objective or the data distribution under the same ckpt lineage.
                "corpus", "balance", "random_caption", "demo_frames", "target_frames",
                "t_sampler", "v_space", "p_drop_text", "p_drop_demo", "p_drop_both",
                "bf16", "warmup_steps", "wd")
        core = ("dim", "depth", "heads", "batch", "lr", "seed", "data_root", "splits_dir",
                "joint_sem", "caption_cache", "texts_json")
        missing = [k for k in core if k not in old_args]
        if missing:
            raise SystemExit(f"[resume] ckpt args missing critical keys {missing} -- a legacy or "
                             f"malformed checkpoint must not bypass config validation")
        # New knobs may be absent in older ckpts; a missing key means the ckpt was trained with
        # the PARSER DEFAULT of its era, so it must be compared against ap.get_default -- comparing
        # against the runtime value would wave through exactly the drift this check exists to catch
        # (legacy ckpt + new flag => missing key silently "equals" the new flag).
        bad = [k for k in crit if old_args.get(k, ap.get_default(k)) != getattr(a, k)]
        if bad:
            raise SystemExit(f"[resume] config mismatch on {bad}: "
                             f"ckpt={[old_args.get(k, ap.get_default(k)) for k in bad]} "
                             f"vs now={[getattr(a, k) for k in bad]}"
                             f" -- refusing silent drift (change the ckpt or the flags)")
        raw_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best_val = ck["epoch"] + 1, ck.get("best_val", float("inf"))
        gstep = ck.get("gstep", 0)
        rng = ck.get("rng")
        # Only rank 0 restores the saved RNG: the ckpt carries ONE stream, and loading it on every
        # rank would make all ranks draw IDENTICAL noise/t after a DDP resume. Non-main ranks keep
        # their startup seeding (a.seed + rank) -- deterministic and rank-distinct.
        if rng is not None and (not ddp or is_main):
            torch.set_rng_state(rng["cpu"]); torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["np"])
        elif ddp and not is_main:
            rng = None
        if is_main:
            print(f"[resume] {a.resume} -> epoch {start_ep}, best_val {best_val:.5f} "
                  f"(rng {'restored' if rng else 'fresh'})", flush=True)
    if is_main:
        (out / "args.json").write_text(json.dumps({**vars(a), "params": n_par,
                                                   "gammas": KIMODO_GAMMAS,
                                                   **({"world_size": int(os.environ["WORLD_SIZE"])}
                                                      if ddp else {})},
                                                  indent=2))
        if ddp:
            print(f"[train] {n_par/1e6:.2f}M params | T={a.demo_frames}+{a.target_frames} | "
                  f"B{a.batch}x{os.environ['WORLD_SIZE']} lr{a.lr} | "
                  f"{len(dl_tr)} steps/epoch/rank", flush=True)
        else:
            print(f"[train] {n_par/1e6:.2f}M params | T={a.demo_frames}+{a.target_frames} | "
                  f"B{a.batch} lr{a.lr} | {len(dl_tr)} steps/epoch", flush=True)

    # ---------------- loop ----------------
    def apply_cfg_drops(b):
        """Per-sample conditioning dropout (F5-style: independent text/demo + joint). Dropped text
        becomes a zero vector (text_mlp's bias path acts as the learned null); a dropped demo is
        zeroed AND masked out of frame_valid so attention cannot see it. valid is rebuilt."""
        if not (a.p_drop_text or a.p_drop_demo or a.p_drop_both):
            return b
        B = b["x"].shape[0]
        db = torch.rand(B, device=dev) < a.p_drop_both
        dt = (torch.rand(B, device=dev) < a.p_drop_text) | db
        dm = (torch.rand(B, device=dev) < a.p_drop_demo) | db
        if dt.any() and "text" in b:
            b["text"] = b["text"].clone(); b["text"][dt] = 0
        if dm.any():
            b["x"] = b["x"].clone(); b["frame_valid"] = b["frame_valid"].clone()
            b["x"][dm, :a.demo_frames] = 0
            b["frame_valid"][dm, :a.demo_frames] = False
            b["valid"] = b["frame_valid"][:, :, None] & b["joint_valid"][:, None, :]
        return b

    def run_validation(ep):
        """rank-0 val + probe + best/last ckpt. Callable from the epoch boundary or mid-epoch
        (step cadence); barrier counts are 2/2 on both sides either way."""
        nonlocal best_val
        if ddp:
            dist.barrier()
        if not is_main:
            if ddp:
                dist.barrier()
            return
        model.eval(); reset_val_stream(ds_va)
        # Fixed captions for val: rotation would make the val text a moving target across passes.
        # ds_va shares `base`; the toggle is safe because the val loader is num_workers=0.
        rc_saved = base.random_caption
        base.random_caption = False
        vtot, vn = 0.0, 0
        with fixed_torch_rng(10_000 + a.seed):
            with torch.no_grad():
                for vb in dl_va:
                    vb = to_dev(vb, dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                        # Same objective as training (t_sampler/v_space) so val tracks what is
                        # optimized; fixed_torch_rng keeps the draws identical across passes.
                        vloss = cfm_loss(model, vb["x"], is_target=vb["is_target"], valid=vb["valid"],
                                         gammas=KIMODO_GAMMAS, t_sampler=a.t_sampler,
                                         v_space=a.v_space, **cond_of(vb))
                    vtot += float(vloss); vn += 1
                    if a.val_max_batches and vn >= a.val_max_batches:
                        break
            v = vtot / max(vn, 1)
            reset_val_stream(ds_va)
            probe_b = to_dev(next(iter(dl_va)), dev)
            p5 = connectivity_probe(raw_model, probe_b, a.demo_frames)
        base.random_caption = rc_saved
        print(f"  [val] flow_loss={v:.5f} | g{gstep} | P5 demo={p5['demo']:.2e} "
              f"text={p5['text']:.2e} sem={p5['joint_sem']:.2e}", flush=True)
        rng_state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                     "np": np.random.get_state()}
        state = {"model": raw_model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                 "gstep": gstep, "val": v, "best_val": min(best_val, v), "args": vars(a),
                 "rng": rng_state}
        atomic_save(state, out / "last_model.pt")
        if v < best_val:
            best_val = v
            atomic_save(state, out / "best_model.pt")
            print(f"  [ckpt] new best val_flow={v:.5f}", flush=True)
        model.train()
        if ddp:
            dist.barrier()

    nonfinite = 0
    for ep in range(start_ep, a.epochs):
        if tr_sampler is not None:
            tr_sampler.set_epoch(ep)
        model.train(); t0, tot, n = time.time(), 0.0, 0
        g_sum, g_max = 0.0, 0.0
        for b in dl_tr:
            b = apply_cfg_drops(to_dev(b, dev))
            # lr warmup (manual: resumes correctly from the saved gstep, no scheduler state)
            if a.warmup_steps > 0:
                lr_now = a.lr * min(1.0, (gstep + 1) / a.warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.bf16):
                loss = cfm_loss(model, b["x"], is_target=b["is_target"], valid=b["valid"],
                                gammas=KIMODO_GAMMAS, t_sampler=a.t_sampler, v_space=a.v_space,
                                **cond_of(b))
            bad = (~torch.isfinite(loss.detach())).float()
            if ddp:
                # The skip decision must be COLLECTIVE: one rank skipping backward while its peers
                # run it deadlocks the DDP reducer at the next bucket sync. MAX-reduce the flag so
                # every rank skips together whenever any rank saw a non-finite loss.
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bad.item() > 0:
                # STABILITY GUARD: skip the step loudly; a silent NaN would poison the weights and
                # every ckpt after it. Abort the run if it becomes a pattern.
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                if is_main:
                    print(f"[WARN] non-finite loss at g{gstep} ep{ep} (#{nonfinite}) -- step "
                          f"skipped on ALL ranks", flush=True)
                if nonfinite >= 50:
                    raise SystemExit("[FATAL] 50 non-finite losses -- training unstable, aborting")
                gstep += 1
                continue
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Gradient overflow can be non-finite even when the loss was finite (bf16 backward).
            # The decision must again be COLLECTIVE under DDP.
            gbad = (~torch.isfinite(gn)).float().to(dev)
            if ddp:
                dist.all_reduce(gbad, op=dist.ReduceOp.MAX)
            if gbad.item() > 0:
                nonfinite += 1
                opt.zero_grad(set_to_none=True)
                if is_main:
                    print(f"[WARN] non-finite GRAD at g{gstep} ep{ep} (#{nonfinite}) -- step "
                          f"skipped on ALL ranks", flush=True)
                if nonfinite >= 50:
                    raise SystemExit("[FATAL] 50 non-finite events -- training unstable, aborting")
                gstep += 1
                continue
            opt.step()
            gnf = float(gn)
            g_sum += gnf; g_max = max(g_max, gnf)
            tot += float(loss.detach()); n += 1
            gstep += 1
            if is_main and gstep % 200 == 0:
                print(f"[g{gstep}] ep{ep} loss={float(loss.detach()):.4f} grad={gnf:.3f} "
                      f"lr={opt.param_groups[0]['lr']:.2e}", flush=True)
            if a.val_every_steps > 0 and gstep % a.val_every_steps == 0:
                run_validation(ep)
            if a.ckpt_snapshot_steps > 0 and gstep % a.ckpt_snapshot_steps == 0 and is_main:
                atomic_save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                             "epoch": ep, "gstep": gstep, "best_val": best_val, "args": vars(a),
                             "rng": {"cpu": torch.get_rng_state(),
                                     "cuda": torch.cuda.get_rng_state_all(),
                                     "np": np.random.get_state()}},
                            out / f"g{gstep:07d}_model.pt")
        if ddp:
            # g_sum rides in the SAME reduction so the printed mean divides a GLOBAL sum by the
            # GLOBAL step count (a local g_sum over a global n would understate the mean 4x).
            agg = torch.tensor([tot, float(n), g_sum], device=dev)
            dist.all_reduce(agg)
            gmax_t = torch.tensor([g_max], device=dev)
            dist.all_reduce(gmax_t, op=dist.ReduceOp.MAX)
            tot, n, g_sum, g_max = float(agg[0]), int(agg[1]), float(agg[2]), float(gmax_t[0])
        if is_main:
            print(f"=== epoch {ep} done in {time.time()-t0:.1f}s | train_flow={tot/max(n,1):.5f} "
                  f"| grad mean={g_sum/max(n,1):.3f} max={g_max:.3f} ===", flush=True)

        if a.val_every_steps == 0 and ((ep + 1) % a.val_every == 0 or ep == a.epochs - 1):
            run_validation(ep)
        # RESUME POLICY: resume always restarts at epoch ck["epoch"]+1. For a MID-epoch snapshot
        # that discards the remainder of the interrupted epoch -- statistically harmless here
        # because draws are random (balanced or shuffled), while gstep/lr/warmup continue exactly.
        if (ep + 1) % a.ckpt_every == 0 and is_main:
            atomic_save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                         "epoch": ep, "gstep": gstep, "best_val": best_val, "args": vars(a),
                         "rng": {"cpu": torch.get_rng_state(),
                                 "cuda": torch.cuda.get_rng_state_all(),
                                 "np": np.random.get_state()}},
                        out / f"ep{ep+1:04d}_model.pt")
    if is_main:
        print("=== training loop complete ===", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
