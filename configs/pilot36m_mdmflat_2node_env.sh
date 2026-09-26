#!/bin/bash
# ADAPTED BASELINE -- a fixed-skeleton text-to-motion denoiser brought to variable skeletons the way every
# comparable work brings one: AnyTop's MDM* ("we concatenate all joint features for each character, and pad them to
# a length of J x D" + "the vectorized rest-pose embedding along the temporal axis as frame 0"), SAMoR's K=1 flat
# tokenizer ("a single flat per-frame feature vector of dimension Jmax x 6", "zero-padded to Jmax=96 joints, with
# padded positions masked", trained "on the same unified heterogeneous corpus ... with the same data splits, loss
# functions, and training budget"), and OmniZoo's MoMask/MMM ("joint padding and binary masking, and append species
# tags to text prompts"; our captions already name the species, so no tag is appended).
#
# Built by SOURCING the simplified baseline so the lineage is shared, then changing exactly what this arm is:
#   * the denoiser is src/models/v2/dit_flat.py -- one token per FRAME holding all joints' channels zero-padded to
#     102 (the corpus maximum), no per-joint tokens, no joint descriptions, no geodesic or directional attention
#     bias, no structural features;
#   * depth 12 instead of 8, so the baseline carries 38.02M parameters against the control's 36.28M -- it is given
#     MORE capacity, not less;
#   * the CONTROL's calibrated group weights: the kimodo gammas are bit-identical in the control's b16 and b32
#     artifacts and in this one (they come from data energies, not from the model), so the baseline optimises the
#     control's objective, which is what "the same loss functions" means.
# Everything else -- corpus, exclusion cut, split, captions, rest-pose demonstration frame, flow-matching
# objective, learning rate, schedule, partition and the frozen evaluation -- is the control's. The BUDGET is the
# attribution arms': 120 epochs against the control's 500-epoch run, with the same 40-epoch decay horizon in
# both, and the comparison is read at the matched epochs 50/75/100 exactly as every other arm's is
# (codex 2026-09-10 #2).
_flat_out=${OUT:-}; _flat_port=${RDZV_PORT:-}
_flat_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=pilot36m_baseline_2node_env.sh
source "$_flat_here/pilot36m_baseline_2node_env.sh"

export DEPTH=12                                  # 38.02M against the control's 36.28M (dim/heads unchanged)
export STRUCT_FEATS=0 DIR_BIAS=0                 # no skeleton inputs, as the flat denoiser has no joint axis
# torch.compile OFF for this arm only. Inductor cannot compile the flat graph under dynamic shapes -- it puts a
# symbolic int in the graph's output list and dies with "AttributeError: 'int' object has no attribute 'meta'"
# (three 8-rank smokes, 2026-09-10; F.pad instead of allocate-and-assign and flatten/unflatten instead of named
# reshapes did not move it). Compilation changes throughput, not arithmetic, and the flat denoiser reads T tokens
# per sample where the per-joint trunk reads T x J, so it is the cheaper model uncompiled.
export COMPILE=0
export OUT=${_flat_out:-runs/v2_noik_pilot36m_mdmflat}
export RDZV_PORT=${_flat_port:-29539}            # baseline 29537, nodesc 29538, rest 29535, restaug 29536
# the control's calibrated weights (identical gammas to configs/pilot_animal_r1acc_gamma_calibration_b16_v2.json,
# re-measured against the current code, same per-cell statistics d8eb75db)
export CALIB=${FLAT_CALIB:-configs/pilot_animal_nodesc_gamma_calibration_b16_v1.json}
export EXTRA="--flat_joints 102 --no_geo_bias ${EXTRA_APPEND:-}"
