#!/usr/bin/env bash
# Launch one arm of the held-out-topology VQVAE ablation.
#
# A separate launcher on purpose. The existing ones are for the legacy corpus and must keep
# working untouched; bolting protocol flags onto them would make one script serve two experiments
# and the difference would live in whichever arguments someone remembered to pass. Here every
# contract-defining choice is written down in one place and echoed into the log, so a run can be
# described from its own output six weeks later.
#
#   bash scripts/_launch_holdout_vqvae.sh                       # single alloc (world = NGPU)
#   NNODES=4 NODE_RANK=<r> MASTER_ADDR=<host>-ib0 ...           # one rank group per alloc
#
# The semantic/control ablation was cancelled by the user on 2026-08-02 ("先不管消融了,我们先
# 紧着泛化性能力做出来"). This launcher runs the semantic arm and refuses anything else.
set -euo pipefail

# The ablation was cancelled: there is ONE run. ARM survives only so that an old command line
# naming the cancelled control arm is refused rather than quietly launching something else.
ARM="${ARM:-semantic}"
[ "$ARM" = "semantic" ] || { echo "REFUSED: ARM='$ARM'. The semantic/control ablation was" >&2
                             echo "         cancelled; this launcher runs the semantic arm only." >&2; exit 2; }

# What this experiment IS, pinned rather than described. Recording the hash of whichever file was
# selected proves only that a file was hashed; it does not prove the run used the pre-registered
# one. Any mismatch below is fatal, so an environment override that points at a different corpus,
# split, freeze or semantic table cannot produce internally-honest provenance for the wrong
# experiment.
EXPECT_SEM_SHA=7cd3f87ebf15b8113d1d36df2893747bd6821fd88caf96458c9c8cd087699222
EXPECT_ART_BODY=0baf7bcfb82266d504f9bb45d0ec4f22980043ee49e53c0d7d13b40ebc858e0c
EXPECT_TRAIN_SHA=6cbd672f886e98b3935eebff17d340e1bc53cb9843852fb9fa944c4ff1d5ae64
EXPECT_VAL_SHA=5798db9db7bba383fcd76ed78fcf3a04244a388a987dbfc12f64b648383d5feb
# The global batch is the scientific quantity; the per-rank batch is an artefact of how many cards
# happen to be available. Pinning per-rank batch instead is what let a 2-GPU run default to global
# 32 against a reference recipe of 64 -- a silent rescaling that only operator memory would catch.
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
DATA_ROOT="${DATA_ROOT:-data/animo4d_L4TB_plus_human_v4b272neutral}"
SPLITS_DIR="${SPLITS_DIR:-data/holdout_splits_v1}"
ARTIFACT="${ARTIFACT:-data/holdout_topologies_v1.json}"
SEMANTICS="${SEMANTICS:-data/joint_semantics_llm2vec_v1.npz}"
EPOCHS="${EPOCHS:-220}"
BATCH=""            # derived from GLOBAL_BATCH and the world size below
LR="${LR:-6.65e-5}"
NGPU="${NGPU:-2}"
# Cross-alloc DDP. NNODES=1 (default) keeps the single-alloc standalone path exactly as it was.
# NNODES>1 takes the static-rendezvous branch, which is required rather than optional here: with
# several Slurm allocations on ONE physical node, c10d's host election compares hostnames, and the
# agent's hostname (swarmh1001) never equals the IB rendezvous host (swarmh1001-ib0), so no rank
# ever starts the TCPStore and every rank sits as a client until it times out. Same-node but
# cross-cgroup also means Slurm isolates GPU P2P and SHM, so NCCL must be told to use IB.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29531}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$NGPU}"
OUT="${OUT:-runs/holdout_vqvae_${ARM}_v1}"

SEM_ON=1     # the ablation was cancelled; this launcher runs the semantic arm only

cd "$(dirname "$0")/.."

# Extra arguments may not re-specify anything that defines the experiment. Without this, appending
# `--protocol legacy` to an otherwise strict command silently downgrades the run: argparse keeps
# the LAST occurrence, so the protected value above would lose to the trailing one, and the run
# would look strict in this file while being legacy in fact.
PROTECTED=(--protocol --splits_dir --holdout_artifact --holdout_sha --joint_semantics
           --semantic_enabled --anytop_root --seed)
# Prefix matching, not equality: `--proto legacy` is an ABBREVIATION of `--protocol` and argparse
# resolves it, so an equality guard lets the exact downgrade it was written to stop walk straight
# through. The trainer also sets allow_abbrev=False; this is the second layer, because a guard
# that depends on a flag in another file is a guard with a remote single point of failure.
for a in "$@"; do
  akey="${a%%=*}"
  for pflag in "${PROTECTED[@]}"; do
    if [[ "$akey" == "$pflag" || ( "$akey" == --?* && "$pflag" == "$akey"* ) ]]; then
      echo "REFUSED: '$pflag' defines this experiment and is fixed by this launcher; passing it" >&2
      echo "         again would override the protected value (argparse keeps the last one)." >&2
      exit 2
    fi
  done
done

WORLD=$(( NNODES * NPROC_PER_NODE ))
if [ $(( GLOBAL_BATCH % WORLD )) -ne 0 ]; then
  echo "REFUSED: global batch $GLOBAL_BATCH is not divisible by world size $WORLD, so no per-rank" >&2
  echo "         batch reproduces the reference recipe. Change the card count or say so explicitly." >&2
  exit 2
fi
BATCH=$(( GLOBAL_BATCH / WORLD ))

# Refuse to start on a codebase whose protected baseline or sealed protocol has drifted. A
# multi-day run launched on top of an accidental edit is a multi-day run thrown away.
python3 scripts/_protect_baseline.py --verify
python3 scripts/_seal_protocol.py --verify --strict-code
python3 scripts/_regression_suite.py

# The expected hash is read from the SEALED copy under protocol/, not from the artifact being
# checked: deriving the expectation from the file it validates verifies nothing. It is the BODY
# hash (the artifact's self-hash), not the file hash — they differ, and passing the wrong one
# aborts every strict run at startup.
ART_SHA="$(python3 -c "
import json,sys
d=json.load(open('protocol/holdout_topologies_v1.json'))
print(d['artifact_sha256'])")"
SEM_SHA="$(python3 -c "import sys;sys.path.insert(0,'.');from src.data import provenance as p;print(p.sha256_file('$SEMANTICS'))")"
TRAIN_SHA="$(python3 -c "import sys;sys.path.insert(0,'.');from src.data import provenance as p;print(p.sha256_file('$SPLITS_DIR/train.txt'))")"
VAL_SHA="$(python3 -c "import sys;sys.path.insert(0,'.');from src.data import provenance as p;print(p.sha256_file('$SPLITS_DIR/val.txt'))")"

# Compare, do not merely record. Every one of these is reachable through an environment variable,
# so without this an operator could point the run at a 768-D table, a different clean split or
# another freeze and get provenance that is internally consistent and describes the wrong
# experiment.
check_sha() {  # name expected actual
  [ "$2" = "$3" ] && return 0
  echo "REFUSED: $1 does not match the pre-registered value." >&2
  echo "         expected $2" >&2
  echo "         actual   $3" >&2
  echo "         This run would carry honest provenance for a different experiment." >&2
  exit 4
}
check_sha "the joint-semantics table"     "$EXPECT_SEM_SHA"   "$SEM_SHA"
check_sha "the held-out artifact body"    "$EXPECT_ART_BODY"  "$ART_SHA"
check_sha "the retained train split"      "$EXPECT_TRAIN_SHA" "$TRAIN_SHA"
check_sha "the retained val split"        "$EXPECT_VAL_SHA"   "$VAL_SHA"

echo "=================================================================="
echo " held-out semantic VQVAE"
echo "   data_root   : $DATA_ROOT"
echo "   splits_dir  : $SPLITS_DIR   (train.txt/val.txt ARE the retained lists)"
echo "   artifact    : $ARTIFACT"
echo "   artifact body sha (from the sealed copy): $ART_SHA"
echo "   semantics   : $SEMANTICS   sha: ${SEM_SHA:-<none>}"
echo "   semantic_enabled: $SEM_ON"
echo "   train/val split sha: ${TRAIN_SHA:0:16} / ${VAL_SHA:0:16}  (match pre-registration)"
echo "   epochs=$EPOCHS  global_batch=$GLOBAL_BATCH = $BATCH x $WORLD ranks  lr=$LR"
echo "   out         : $OUT"
echo "=================================================================="

# --holdout_sha is passed explicitly rather than derived, so a swapped artifact aborts instead of
# being silently adopted: deriving the expectation from the same file it checks would verify
# nothing.
if [ "$NNODES" -gt 1 ]; then
  [ -z "$MASTER_ADDR" ] && { echo "REFUSED: NNODES>1 requires MASTER_ADDR (use the IB name, e.g." >&2
                             echo "         swarmh1001-ib0; the plain hostname does not resolve to" >&2
                             echo "         the fabric the ranks actually talk over)." >&2; exit 2; }
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
  export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  RDZV=(--nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR"
        --master_port="$MASTER_PORT" --nproc_per_node="$NPROC_PER_NODE")
  echo "   rendezvous : static, node_rank=$NODE_RANK/$NNODES via $MASTER_ADDR:$MASTER_PORT"
  echo "   world size : $(( NNODES * NPROC_PER_NODE ))   global batch = $(( BATCH * NNODES * NPROC_PER_NODE ))"
else
  RDZV=(--standalone --nproc_per_node="$NGPU")
fi

exec torchrun "${RDZV[@]}" scripts/train_graph_vqvae.py \
  --anytop_root "$DATA_ROOT" \
  --splits_dir "$SPLITS_DIR" \
  --protocol unseen_topology_v1 \
  --holdout_artifact "$ARTIFACT" \
  --holdout_sha "$ART_SHA" \
  --joint_semantics "$SEMANTICS" \
  --semantic_enabled "$SEM_ON" \
  --max_joints 144 --max_coarse 96 --max_frames 64 \
  --d_model 512 --n_heads 8 --d_ff 1536 \
  --n_graph_layers 4 --n_enc_temporal_layers 2 \
  --n_pre_vq_layers 2 --n_post_vq_layers 2 --n_cross_layers 3 --n_dec_temporal_layers 2 \
  --temporal_stride 4 --temporal_kernel 9 \
  --code_dim 512 --num_codes 8192 --num_quantizers 4 \
  --quantize_dropout_prob 0.1 --ema_mu 0.99 --dead_code_threshold 1.0 \
  --w_pos 1.0 --w_rot 1.0 --w_vel 1.0 --w_contact 0.1 --w_world 0.25 --w_fk 1.0 \
  --w_traj 0.1 --w_commit 0.02 \
  --human_upsample_factor 3.0 --human_upsample_start_epoch 0 \
  --human_upsample_phase2_factor 4.5 --human_upsample_phase2_start_epoch 50 \
  --epochs "$EPOCHS" --batch_size "$BATCH" --lr "$LR" \
  --amp_dtype bf16 --num_workers 3 --seed 42 \
  --save_every 5 --periodic_save_every 25 --log_every 50 \
  --out "$OUT" "$@"
