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
                                      truebones_types, DEMO_FRAMES, TARGET_FRAMES)
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


def connectivity_probe(model, b):
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
    return {"demo": float(g_x[:, :DEMO_FRAMES].norm()),
            "target": float(g_x[:, DEMO_FRAMES:].norm()),
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
    tb = truebones_types(cond.keys())
    names = {k: read_split(a.splits_dir, k) for k in ("train", "val")}
    base = AnyTopDataset(data_root=a.data_root, split="all", num_frames=300, max_joints=144,
                         load_captions=True, caption_emb_cache=a.caption_cache,
                         random_caption=False, augment=False, joint_semantics=a.joint_sem,
                         species_whitelist=tb, splits_dir=a.splits_dir,
                         texts_json_name=a.texts_json)
    ds_tr = InContextPairs(base, names["train"], names["train"], object_types=tb,
                           balance_skeletons=True, seed=a.seed)
    ds_va = InContextPairs(base, names["val"], names["train"], object_types=tb,
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
        model = DDP(model, device_ids=[local_rank])
    n_par = sum(p.numel() for p in raw_model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    start_ep, best_val = 0, float("inf")
    if a.resume:
        # Resume is a STATISTICAL continuation, not bit-exact: DataLoader worker streams re-derive
        # from a fresh base_seed after restart. What must NOT drift silently is the config -- a
        # resumed run with different dims/lr/data would corrupt the ckpt lineage, so critical
        # fields are hard-checked. Torch/numpy RNG states are restored best-effort on top.
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)  # our own ckpt; contains numpy RNG state, rejected by 2.6's weights_only default
        old_args = ck.get("args", {})
        crit = ("dim", "depth", "heads", "batch", "lr", "seed", "data_root", "splits_dir",
                "joint_sem", "caption_cache", "texts_json")
        missing = [k for k in crit if k not in old_args]
        if missing:
            raise SystemExit(f"[resume] ckpt args missing critical keys {missing} -- a legacy or "
                             f"malformed checkpoint must not bypass config validation")
        bad = [k for k in crit if old_args[k] != getattr(a, k)]
        if bad:
            raise SystemExit(f"[resume] config mismatch on {bad}: "
                             f"ckpt={[old_args[k] for k in bad]} vs now={[getattr(a, k) for k in bad]}"
                             f" -- refusing silent drift (change the ckpt or the flags)")
        raw_model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep, best_val = ck["epoch"] + 1, ck.get("best_val", float("inf"))
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
                                                   "demo_frames": DEMO_FRAMES,
                                                   "target_frames": TARGET_FRAMES,
                                                   **({"world_size": int(os.environ["WORLD_SIZE"])}
                                                      if ddp else {})},
                                                  indent=2))
        if ddp:
            print(f"[train] {n_par/1e6:.2f}M params | T={DEMO_FRAMES}+{TARGET_FRAMES} | "
                  f"B{a.batch}x{os.environ['WORLD_SIZE']} lr{a.lr} | "
                  f"{len(dl_tr)} steps/epoch/rank", flush=True)
        else:
            print(f"[train] {n_par/1e6:.2f}M params | T={DEMO_FRAMES}+{TARGET_FRAMES} | "
                  f"B{a.batch} lr{a.lr} | {len(dl_tr)} steps/epoch", flush=True)

    # ---------------- loop ----------------
    for ep in range(start_ep, a.epochs):
        if tr_sampler is not None:
            tr_sampler.set_epoch(ep)
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for b in dl_tr:
            b = to_dev(b, dev)
            loss = cfm_loss(model, b["x"], is_target=b["is_target"], valid=b["valid"],
                            gammas=KIMODO_GAMMAS, **cond_of(b))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()); n += 1
        if ddp:
            agg = torch.tensor([tot, float(n)], device=dev)
            dist.all_reduce(agg)
            tot, n = float(agg[0]), int(agg[1])
        if is_main:
            print(f"=== epoch {ep} done in {time.time()-t0:.1f}s | train_flow={tot/max(n,1):.5f} ===",
                  flush=True)

        if (ep + 1) % a.val_every == 0 or ep == a.epochs - 1:
            # Validation, probe, and every checkpoint write are RANK-0 ONLY (cross-alloc rule 5:
            # concurrent writers on the shared fs race). Other ranks hold at the barrier -- val is
            # ~10 s, far inside the NCCL default timeout.
            if ddp:
                dist.barrier()
            if not is_main:
                if ddp:
                    dist.barrier()
                continue
            model.eval(); reset_val_stream(ds_va)
            vtot, vn = 0.0, 0
            with fixed_torch_rng(10_000 + a.seed):
                with torch.no_grad():
                    for b in dl_va:
                        b = to_dev(b, dev)
                        vloss = cfm_loss(model, b["x"], is_target=b["is_target"], valid=b["valid"],
                                         gammas=KIMODO_GAMMAS, **cond_of(b))
                        vtot += float(vloss); vn += 1
                v = vtot / max(vn, 1)
                reset_val_stream(ds_va)
                probe_b = to_dev(next(iter(dl_va)), dev)
                p5 = connectivity_probe(raw_model, probe_b)
            print(f"  [val] flow_loss={v:.5f} | P5 demo={p5['demo']:.2e} "
                  f"text={p5['text']:.2e} sem={p5['joint_sem']:.2e}", flush=True)
            rng_state = {"cpu": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all(),
                         "np": np.random.get_state()}
            state = {"model": raw_model.state_dict(), "opt": opt.state_dict(),
                     "epoch": ep, "val": v, "best_val": min(best_val, v), "args": vars(a),
                     "rng": rng_state}
            atomic_save(state, out / "last_model.pt")
            if v < best_val:
                best_val = v
                atomic_save(state, out / "best_model.pt")
                print(f"  [ckpt] new best val_flow={v:.5f}", flush=True)
            if ddp:
                dist.barrier()
        if (ep + 1) % a.ckpt_every == 0 and is_main:
            atomic_save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                         "epoch": ep, "best_val": best_val, "args": vars(a),
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
