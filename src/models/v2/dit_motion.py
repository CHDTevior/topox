"""v2 prototype: in-context motion DiT with flow matching and an infilling objective.

DESIGN RATIONALE (each choice traces to a measured failure of v1)

  in-context demo, not a learned species prior
      v1 learned P(motion|skeleton) and barely P(motion|skeleton,text): text contributed 4.6% of
      the loss gradient and its cross-attention branch was functionally dead. Root cause: species
      identity explains a 24.8x spread in normalised motion speed, so text could never compete.
      Here the demo motion carries "how THIS rig moves" and text only has to pick the action.
      Measured support: static structure does NOT predict a species' motion prior
      (joints r=+0.12, rest-radius r=-0.32), so it must come from examples, not from the skeleton.

  infilling objective
      Self-supervised: mask a span, predict it from the rest plus text. Needs no paired data, and
      the inference-time layout (demo | target) is exactly the training-time layout, so there is no
      train/test mismatch. Borrowed from text-guided speech infilling (F5-TTS / E2-TTS), where the
      same trick gives zero-shot voice cloning from seconds of reference audio.

  factorised spatio-temporal attention
      Naive per-joint-per-frame tokens are 300x144 = 43k, which is intractable. Attention alternates
      over time (length T) and over joints (length J<=144), so each attention stays small.

  flow matching (OT-CFM), x-prediction
      v1's ablation measured x-prediction as ~2.2x better than v-prediction on geometry, and 5 ODE
      steps sufficed at inference (game latency requirement).

  blueprint via AdaLN, not cross-attention
      v1's cross-attention text branch died (step response 0.0000 at every probe strength). A
      low-dimensional blueprint modulating every block through AdaLN cannot be routed around: it
      scales and shifts every activation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint  # activation checkpointing (grad_ckpt)
import torch.nn.functional as F
from src.models.v2.spec_rope import SpectralJointRoPE, apply_rotary_pos_emb
from src.models.v2.temporal_rope import sinusoidal_cos_sin


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    return F.pad(emb, (0, dim - emb.shape[-1])) if dim % 2 else emb


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class Attention(nn.Module):
    """Multi-head self-attention with an optional additive [B,H|1,N,N] bias (used to inject the
    skeleton graph on the joint axis) and an optional key-padding mask.

    qk_norm: RMS-normalise q and k per head before the dot product (ViT-22B / Gemma recipe).
    MEASURED MOTIVE (2026-08-26): without it, blocks[0].t_attn's max logit sat at 1372-1483 on
    HEALTHY run10 checkpoints (a sane transformer runs O(10)) and exploded to 12392-23391 across
    the ep34 damage step. A saturated softmax has near-zero local gradients (which is also why
    Adam's v collapsed on the conditioning biases) yet a tiny parameter move flips its argmax --
    the "ordinary-gradient damage step" that killed seven runs. Normalising q and k bounds the
    logits by construction. Flag-gated so every pre-run11 checkpoint still loads bit-identically."""

    def __init__(self, dim: int, n_heads: int, qk_norm: bool = False):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            # RMSNorm over head_dim with a learnable gain, applied to q AND k: logits become
            # bounded by |g_q||g_k|*dh/sqrt(dh) regardless of how far training sharpens.
            self.q_norm = nn.RMSNorm(self.dh, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.dh, eps=1e-6)

    def forward(self, x, attn_bias=None, key_pad=None, rope=None):
        """x [..., N, D] (any leading batch dims); attn_bias broadcastable to [..., H, N, N];
        key_pad [..., N] True=valid, broadcastable likewise. rope: optional (cos, sin), each broadcastable to
        q / k [..., H, N, dh] -- the spectral joint RoPE, applied AFTER the q/k normalisation as UniMate does.

        NOTHING is expanded or cloned (codex 01a01b1a fix A): the old path materialised the bias
        at [B*T, H, N, N] three times over (repeat_interleave + .clone() + key-mask add) --
        ~0.75 GiB fp32 at TrueBones scale, ~1.57 GiB at the 262M config. SDPA broadcasts the
        mask itself; callers pass compact shapes like [B, 1, H, J, J] against x [B, T, J, D].
        Callers must NOT pass both attn_bias and key_pad with shapes whose sum would broadcast-
        materialise (the spatial path encodes padding inside joint_bias's PAD_BIAS instead)."""
        N, D = x.shape[-2], x.shape[-1]
        lead = x.shape[:-2]
        qkv = self.qkv(x).reshape(*lead, N, 3, self.h, self.dh)
        q, k, v = (t.transpose(-3, -2) for t in qkv.movedim(-3, 0))   # each [..., H, N, dh]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rotary_pos_emb(q, k, rope[0], rope[1])
        bias = attn_bias
        if key_pad is not None:                       # [..., N] True = valid
            m = torch.zeros(*key_pad.shape[:-1], 1, 1, N, device=x.device, dtype=q.dtype)
            m = m.masked_fill(~key_pad[..., None, None, :], torch.finfo(q.dtype).min)
            bias = m if bias is None else bias + m
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(o.transpose(-3, -2).reshape(*lead, N, D))


class Block(nn.Module):
    """One factorised block: temporal attention -> spatial (joint) attention -> MLP,
    every sub-layer AdaLN-modulated by the conditioning vector c."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0, qk_norm: bool = False):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(3))
        self.t_attn = Attention(dim, n_heads, qk_norm=qk_norm)
        self.s_attn = Attention(dim, n_heads, qk_norm=qk_norm)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(approximate="tanh"), nn.Linear(h, dim))
        # zero-init so an untrained block is the identity: pretrained-style stability, and the
        # conditioning starts as a no-op rather than as noise.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        # Zero the WEIGHTS so conditioning starts as a no-op, but bias the three GATES to 1 so every
        # block starts as an ordinary residual transformer.
        # Why this matters (measured): with gates zero-initialised too, gate grads were ~3e-06, so at
        # lr 1e-3 they move ~1e-5 over 3000 steps -- they never leave zero. x = x + gate*attn(...)
        # with gate~0 makes every block an exact identity, collapsing the whole network to
        # out(LayerNorm(Linear(x))), a linear map. It could not fit even ONE clip
        # (sanity rung L1: loss plateaued at 46% of the target variance, sampling relL2 0.588,
        # and relL2 got WORSE with more ODE steps: 1.01 / 1.20 / 1.23 / 1.28 at 1/5/20/100 steps).
        # The modulation weight must NOT be zero. With W=0 the AdaLN output is exactly the bias,
        # so shift/scale are constants and the conditioning c never reaches any activation --
        # measured: a one-hot condition of magnitude 10 produced a conditioning effect of 0.009,
        # i.e. nothing, and restricting t to [0,0.05] left the loss at 0.617 against a
        # predict-the-mean baseline of 0.621. Small non-zero init gives conditioning a live path
        # from step 1; gate bias 1 keeps each block an ordinary residual at initialisation.
        nn.init.normal_(self.ada[-1].weight, std=0.5); nn.init.zeros_(self.ada[-1].bias)
        with torch.no_grad():                       # gate slots are chunks 2, 5, 8 of 9
            for k in (2, 5, 8):
                self.ada[-1].bias[k * dim:(k + 1) * dim].fill_(1.0)

    def forward(self, x, c, joint_bias=None, frame_valid=None, joint_valid=None, rope_cos=None, rope_sin=None,
                trope_cos=None, trope_sin=None):
        # x [B,T,J,D]; rope_cos / rope_sin [B,1,1,J,dh] (spectral joint RoPE) reach the spatial attention,
        # trope_cos / trope_sin [1,1,T,dh] (sinusoidal temporal RoPE) the temporal one -- positional tensors, not
        # tuples, so the activation-checkpoint call below can pass them through
        B, T, J, D = x.shape
        p = self.ada(c)                                    # [B, 9D]
        (st, sc, sg, st2, sc2, sg2, mt, mc, mg) = p.chunk(9, dim=-1)
        e = lambda v: v[:, None, None, :]                  # broadcast over T,J

        # --- temporal: tokens are frames, batched over joints ---
        y = modulate(self.n1(x), e(st), e(sc)).permute(0, 2, 1, 3).reshape(B * J, T, D)
        fv = frame_valid.repeat_interleave(J, 0) if frame_valid is not None else None
        y = self.t_attn(y, key_pad=fv, rope=(trope_cos, trope_sin) if trope_cos is not None else None
                        ).reshape(B, J, T, D).permute(0, 2, 1, 3)
        x = x + e(sg) * y

        # --- spatial: tokens are joints, batched over frames; skeleton graph enters as bias ---
        # x stays [B,T,J,D]: SDPA broadcasts the compact bias over T instead of the old
        # repeat_interleave + clone materialisation (codex 01a01b1a fix A). Joint padding is NOT
        # passed as key_pad here: collate already writes PAD_BIAS -1e4 into every padded row and
        # column of joint_bias, which zeroes padded keys in softmax exactly like the old -inf
        # mask did (exp(-1e4) underflows to 0), without a second broadcast-materialising add.
        y = modulate(self.n2(x), e(st2), e(sc2))
        jb = joint_bias
        if jb is not None:
            jb = jb[:, None, None] if jb.dim() == 3 else jb[:, None]   # -> [B,1,1|H,J,J]
        jv = None if jb is not None else \
            (joint_valid[:, None] if joint_valid is not None else None)  # only if no bias given
        y = self.s_attn(y, attn_bias=jb, key_pad=jv, rope=(rope_cos, rope_sin) if rope_cos is not None else None)
        x = x + e(sg2) * y

        x = x + e(mg) * self.mlp(modulate(self.n3(x), e(mt), e(mc)))
        return x


class InContextMotionDiT(nn.Module):
    """[demo | target] motion tokens -> velocity field, conditioned on (t, text, blueprint, skeleton).

    Forward takes the FULL sequence (demo frames followed by target frames). Demo frames are given
    clean; target frames are the flow-matching interpolant. `is_target` marks which frames the loss
    is taken on. That single layout serves both training (random span) and inference (demo | noise).
    """

    def __init__(self, in_ch=13, dim=256, depth=6, n_heads=8, d_text=4096, d_blueprint=16,
                 d_joint_sem=4096, mlp_ratio=4.0, use_struct_feats=False, use_dir_bias=False,
                 local_root_dim=0, use_ref_text=False, grad_ckpt=False, qk_norm=False, use_geo_bias=True,
                 use_spec_rope=False, spec_rope_k=8, spec_rope_hks=False, use_temporal_rope=False, trope_base=700.0,
                 struct_world_rest=False):
        super().__init__()
        # Activation checkpointing recomputes each block's activations in the backward pass
        # instead of storing them: ~60-70% less activation memory for ~30% more compute. It is
        # what makes a wider/deeper trunk fit while KEEPING global batch 32 -- shrinking the batch
        # instead would worsen the tail-sample-count problem that drives the instability.
        # Inference-only paths (torch.no_grad) fall through to the plain call.
        self.grad_ckpt = bool(grad_ckpt)
        self.dim = dim
        self.n_heads = n_heads
        self.x_in = nn.Linear(in_ch, dim)
        # graph-v2 flags; the MODULES are constructed at the very END of __init__ so that their
        # parameter initialisation consumes RNG *after* every baseline tensor is drawn -- with
        # them here, the same seed produced different backbone weights per flag combination
        # (codex 01a01b1a fix B), silently confounding any cross-arm comparison.
        self.use_struct_feats = bool(use_struct_feats)
        # R1/R2 (2026-09-25): the struct features carry a 6-d world-frame rest descriptor (bone direction at rest, rest
        # position; src/data/incontext_pairs.py _world_rest_feats) -- a choice WITHIN the struct-feats arm. The extra
        # columns enter through their own bias-free Linear (struct_rest_in, built at the very END of __init__ so every
        # other tensor is drawn from the same RNG sequence as the 8-column arm); its output is added to the first
        # struct_mlp layer's pre-activation. The checkpoint states the flag by that parameter: a strict load refuses
        # 14-column weights in an 8-column model (unexpected key) and the reverse (missing key).
        self.struct_world_rest = bool(struct_world_rest)
        if self.struct_world_rest and not self.use_struct_feats:
            raise ValueError("struct_world_rest extends the struct_feats table: it needs use_struct_feats=True")
        self.struct_in = 8 + (6 if self.struct_world_rest else 0)
        self.use_dir_bias = bool(use_dir_bias)
        self.use_geo_bias = bool(use_geo_bias)
        # UniMate's spectral joint RoPE (src/models/v2/spec_rope.py; user 2026-09-15 "照 UniMate: 谱 RoPE 替掉 j_pos"): the
        # joint-slot table j_pos is NOT created (nor added) when it is on; the SignNet module is built LAST (see below).
        self.use_spec_rope = bool(use_spec_rope)
        self.spec_rope_k = int(spec_rope_k)
        # H1 (2026-09-24): the spectral RoPE's coordinates are the rig's heat-kernel signature at K scales and its encoder
        # a plain MLP (src/models/v2/spec_rope.py HeatKernelSpectralEncoder) instead of the SignNet on K eigenvectors. A
        # choice WITHIN the spectral arm, not a module of its own: it needs use_spec_rope, and spec_rope_k then counts scales.
        self.spec_rope_hks = bool(spec_rope_hks)
        if self.spec_rope_hks and not self.use_spec_rope:
            raise ValueError("spec_rope_hks needs use_spec_rope=True: the heat-kernel signature is the spectral RoPE's "
                             "coordinate, not a module of its own")
        # Sinusoidal temporal RoPE (src/models/v2/temporal_rope.py; user 2026-09-16), the same trade on the frame axis:
        # the learned t_pos table is NOT created (nor added) when it is on. The base travels with the weights as a
        # persistent buffer, so a consumer that rebuilds the model with the wrong base gets the right one back from the
        # checkpoint, and its presence / absence makes a mismatched rebuild fail strict loading either way.
        self.use_temporal_rope = bool(use_temporal_rope)
        if self.use_temporal_rope:
            # The base is kept BOTH as a python float, which is what the forward reads, and as a persistent buffer,
            # which is what travels with the weights. The float is what torch.compile can specialise on: reading
            # float(buffer) inside the forward broke on the first recompilation (the traced value is not a real
            # number there, and the table builder's own range check raised). The buffer exists so that a consumer
            # rebuilding with the wrong base is REFUSED rather than silently corrected by the checkpoint while its
            # args keep saying the wrong number -- the calibration is bound to the args (codex trope r1 P1-2) -- and
            # the pre-hook below makes the two agree by construction, so the forward may trust the float.
            self.trope_base_value = float(trope_base)
            self._register_load_state_dict_pre_hook(self._refuse_trope_base_drift)
        if not self.use_geo_bias:
            # simplified-baseline arm (user 2026-09-08; codex baseline r1 #1): the spatial attention keeps NO skeleton-graph prior.
            # The flag travels with the checkpoint as a buffer, so a consumer that rebuilds the model without use_geo_bias=False
            # fails strict loading (unexpected key) instead of silently restoring the geodesic bias at evaluation. No RNG consumed.
            self.register_buffer("geo_bias_off", torch.ones(()))
        self.mask_token = nn.Parameter(torch.zeros(dim))          # marks "this frame is to be generated"
        # TEMPORAL POSITION. Without this the model cannot tell frame 1 from frame 16, so when the
        # input is pure noise (t->0) it has no way to know what belongs at each timestep and can only
        # emit the per-frame mean. Measured before adding it: error at t=0 was 0.997 (i.e. nothing
        # learned) while error at t=0.7 was 0.024 (copying the input, which needs no position), and
        # the conditioning effect sat at 0.010 no matter how the conditioning was scaled -- AdaLN
        # broadcasts one [D] shift/scale to every token, so without positions it cannot paint a
        # T x J x C target that differs per position.
        self.max_T, self.max_J = 4096, 160
        t_pos = torch.zeros(1, self.max_T, 1, dim)      # a Parameter only when the learned table is in use (below)
        # JOINT POSITION, separate from joint SEMANTICS. Semantics say what a joint IS
        # ("the left claw of the arm"); position says which slot it occupies. They are complementary:
        # several joints down one limb have near-identical descriptions, and joint_semantics is
        # optional, so without an index embedding the model cannot address individual joints at all.
        # Same failure as the missing temporal position: fine when copying the input, useless when
        # generating from noise.
        # With use_spec_rope the slot table is replaced by the spectral RoPE: the same normal draw still happens (into a
        # tensor that is then dropped) so the backbone parameters after this point are bit-identical under one seed
        # whichever way the flag is set (the causal-pairing discipline of the graph-v2 modules, codex 01a01b1a fix B).
        j_pos = torch.zeros(1, 1, self.max_J, dim)
        nn.init.normal_(t_pos, std=0.02); nn.init.normal_(j_pos, std=0.02)
        if not self.use_temporal_rope:
            self.t_pos = nn.Parameter(t_pos)
        else:
            self.register_buffer("trope_base", torch.tensor(float(trope_base)))
        if not self.use_spec_rope:
            self.j_pos = nn.Parameter(j_pos)
        self.joint_sem = nn.Linear(d_joint_sem, dim)              # per-joint identity (LLM2Vec)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.text_mlp = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        # blueprint is low-dimensional and explicit: it modulates every block and cannot be routed around
        self.bp_mlp = nn.Sequential(nn.Linear(d_blueprint, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, n_heads, mlp_ratio, qk_norm=qk_norm)
                                     for _ in range(depth)])
        self.n_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, in_ch)
        # Zero-init the adaLN modulation ONLY (as in DiT). The output projection must NOT also be
        # zero-initialised: if both are zero the first forward pass is identically 0, so no gradient
        # reaches any AdaLN gate, every block stays an exact identity, and the conditioning can never
        # switch on. Measured directly: with both zeroed, gate grads were 0.00e+00 in every block and
        # the overfit gate G4 (conditioning-vs-noise effect) sat at 1.68x instead of >3x.
        nn.init.normal_(self.ada_out[-1].weight, std=0.5); nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.normal_(self.out.weight, std=0.02); nn.init.zeros_(self.out.bias)

        # ---- graph-v2 modules, LAST so the shared backbone init above is identical across all
        # flag combinations under one seed (codex 01a01b1a fix B) ----
        # Knife 1: structural joint features. j_pos is a bare SLOT table -- slot 17 of Monkey and
        # slot 17 of Elephant share one learned vector, so unseen rigs inherit wrong priors. The
        # 8-d structural descriptor (rest offset direction, bone length, depth, child count, leaf
        # flag) is computed from the skeleton itself and generalises by construction. The FINAL
        # projection is zero-initialised: every arm starts as the exact baseline function and the
        # feature pathway ramps in by gradient, keeping E0/E1/E2/E3 causally comparable.
        if self.use_struct_feats:
            self.struct_mlp = nn.Sequential(nn.Linear(8, dim), nn.SiLU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.struct_mlp[-1].weight)
            nn.init.zeros_(self.struct_mlp[-1].bias)
        # Knife 2: directional learnable per-head attention bias (Graphormer-style). The fixed
        # -clip(geodesic,8) scalar is symmetric (grandparent == grandchild == 2), head-shared and
        # blind past 8 hops. Two zero-initialised tables indexed by (up,down) LCA hop counts are
        # ADDED to the existing scalar bias: at init the model is bit-identical to baseline, and
        # the tables learn per-head, per-direction corrections out to UPDOWN_CLIP=15 hops.
        if self.use_dir_bias:
            self.e_up = nn.Embedding(16, n_heads)
            self.e_down = nn.Embedding(16, n_heads)
            nn.init.zeros_(self.e_up.weight)
            nn.init.zeros_(self.e_down.weight)
        # Two-stage bridge input (Kimodo's local-root block). LAST and zero-initialised, same
        # causal-pairing discipline as the graph-v2 modules. It carries a DETERMINISTIC function
        # of the root tower's prediction, and the body tower's global-root slots are zeroed, so
        # this is a replacement pathway, not an additive shortcut.
        # F5-TTS reference-transcript analogue (2026-08-20, user: "和它对齐"). F5 lays
        # [ref_text + gen_text] over [ref_mel + masked span], so the model is TOLD what the
        # reference says and can factor the reference's CONTENT out, keeping only its style --
        # which is why a few seconds of any utterance clones a voice. Ours is that trick on the
        # [demo | target] layout: ONE shared projection (F5 has a single text encoder), applied
        # PER FRAME -- the demo's own caption over the demo frames, the request over the frames
        # being generated. Position is what LETS the two captions be told apart at the input; it is
        # NOT an information barrier (codex round-S5): temporal attention still carries demo-frame
        # content into target frames, and the global AdaLN target-text condition still modulates
        # every frame. The claim is only that the model can DISTINGUISH "what the reference did"
        # from "what is being asked", which is the factorization the old design withheld outright.
        # Zero-initialised and built LAST: flag off = bit-identical weights under one seed.
        self.use_ref_text = bool(use_ref_text)
        if self.use_ref_text:
            self.ref_text_mlp = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, dim),
                                              nn.SiLU(), nn.Linear(dim, dim))
            nn.init.zeros_(self.ref_text_mlp[-1].weight)
            nn.init.zeros_(self.ref_text_mlp[-1].bias)
        self.local_root_mlp = None
        if local_root_dim > 0:
            self.local_root_mlp = nn.Linear(local_root_dim, dim)
            nn.init.zeros_(self.local_root_mlp.weight)
            nn.init.zeros_(self.local_root_mlp.bias)
        # Spectral joint RoPE, LAST (its phi / rho draws come after every backbone tensor). UniMate's own init: default
        # Linear init for phi and rho, rho's last layer normal(0.02) / zero bias -- small angles at step 0, not zero, as in
        # their code; NOT zero-initialised, so this arm is not the exact baseline function at init (the baseline's j_pos
        # is gone anyway). Its parameters are the checkpoint's marker: a consumer that rebuilds the model without the flag
        # fails strict loading (unexpected spec_rope.* keys, missing j_pos) instead of silently restoring a slot table.
        # Under spec_rope_hks the encoder's keys are the MLP's (spec_rope.spectral_encoder.net.*) plus its `scales` buffer,
        # so a SignNet checkpoint and an HKS checkpoint refuse each other's weights the same way.
        self.spec_rope = None
        if self.use_spec_rope:
            self.spec_rope = SpectralJointRoPE(head_dim=dim // n_heads, num_eigvecs=self.spec_rope_k,
                                               hks=self.spec_rope_hks)
        # R1/R2 world-rest columns (see the flag above): built last, after spec_rope, so the 8-column arm's tensors are
        # bit-identical under the same seed whether or not the flag is on.
        self.struct_rest_in = None
        if self.struct_world_rest:
            self.struct_rest_in = nn.Linear(6, dim, bias=False)
    def _refuse_trope_base_drift(self, state_dict, prefix, *args):
        key = prefix + "trope_base"
        if key in state_dict and float(state_dict[key]) != self.trope_base_value:
            raise RuntimeError(
                f"[refuse] this checkpoint was trained with trope_base={float(state_dict[key])} but the model was built "
                f"with {self.trope_base_value}: the loaded buffer would silently overrule the flag this run records "
                f"and the calibration is bound to the flag, not to the buffer")

    def forward(self, x, t, *, is_target, joint_sem=None, text=None, blueprint=None,
                joint_bias=None, frame_valid=None, joint_valid=None,
                struct_feats=None, updown=None, local_root=None, demo_text=None, spectral_feats=None):
        """x [B,T,J,C] ; t [B] in [0,1] ; is_target [B,T] bool.
        struct_feats [B,J,8] (14 with struct_world_rest) / updown [B,J,J,2] are graph-v2 inputs; both ignored
        (and refused) unless the matching use_* flag built the module.
        spectral_feats [B,J,K] are the rig's Laplacian eigenvectors (src/data/skeleton_spectral.py), or with
        spec_rope_hks its heat-kernel signature at K scales; required by and only accepted with use_spec_rope (a
        mismatch either way is refused).
        local_root [B,T,4] is the two-stage bridge input (Kimodo's local-root representation);
        it is refused unless local_root_dim built the module."""
        B, T, J, _ = x.shape
        h = self.x_in(x)
        trope_cos = trope_sin = None
        if self.use_temporal_rope:
            # float32 table ALWAYS, whatever autocast is doing to h: at bf16 the stored cos/sin miss the unit
            # circle by 5.5e-3 (codex trope r1 P2), and apply_rotary_pos_emb does its arithmetic in float32 anyway
            cs, sn = sinusoidal_cos_sin(T, self.dim // self.n_heads, self.trope_base_value,
                                        device=h.device, dtype=torch.float32)
            trope_cos, trope_sin = cs[None, None], sn[None, None]                # [1,1,T,dh]: over the folded B*J and H
        else:
            h = h + self.t_pos[:, :T]                                           # temporal position (learned table)
        rope_cos = rope_sin = None
        if self.use_spec_rope:
            if spectral_feats is None:
                raise ValueError("model built with use_spec_rope=True but batch has no spectral_feats -- dataset must "
                                 "be built with emit_spectral=K")
            if spectral_feats.shape[-1] != self.spec_rope_k:
                raise ValueError(f"spectral_feats has K={spectral_feats.shape[-1]}, model built with spec_rope_k={self.spec_rope_k}")
            cos, sin = self.spec_rope.cos_sin(spectral_feats)                   # [B,J,dh] each, once per forward
            rope_cos, rope_sin = cos[:, None, None], sin[:, None, None]         # [B,1,1,J,dh]: over T and H
        else:
            if spectral_feats is not None:
                raise ValueError("spectral_feats passed but the model was built with use_spec_rope=False")
            h = h + self.j_pos[:, :, :J]                                        # joint position (slot table)
        h = h + is_target[..., None, None].to(h.dtype) * self.mask_token       # flag frames to generate
        if self.use_ref_text and demo_text is not None:
            tgt_v = self.ref_text_mlp(text if text is not None else torch.zeros_like(demo_text))
            dem_v = self.ref_text_mlp(demo_text)
            per_frame = torch.where(is_target[..., None], tgt_v[:, None], dem_v[:, None])
            h = h + per_frame[:, :, None, :].to(h.dtype)                # [B,T,1,D] -> every joint
        if local_root is not None:
            if self.local_root_mlp is None:
                raise ValueError("local_root passed but the model was built with "
                                 "local_root_dim=0")
            h = h + self.local_root_mlp(local_root.to(h.dtype))[:, :, None, :]  # [B,T,1,D]->all J
        if joint_sem is not None:
            h = h + self.joint_sem(joint_sem)[:, None]                          # [B,1,J,D]
        if self.use_struct_feats:
            if struct_feats is None:
                raise ValueError("model built with use_struct_feats=True but batch has no "
                                 "struct_feats -- dataset must be built with emit_graph_v2=True")
            if struct_feats.shape[-1] != self.struct_in:
                raise ValueError(f"struct_feats has {struct_feats.shape[-1]} columns but the model was built for "
                                 f"{self.struct_in} (struct_world_rest={self.struct_world_rest}): dataset and model disagree")
            z = self.struct_mlp[0](struct_feats[..., :8])
            if self.struct_world_rest:
                z = z + self.struct_rest_in(struct_feats[..., 8:])
            h = h + self.struct_mlp[2](self.struct_mlp[1](z))[:, None]         # [B,1,J,D]
        if not self.use_geo_bias and joint_bias is not None:
            # drop the geodesic entries (-clip(geo, 8), in [-8, 0]) but keep the PAD_BIAS (-1e4) padding entries the collator
            # wrote, so padded joints stay masked exactly as before
            joint_bias = torch.where(joint_bias <= -1e3, joint_bias, torch.zeros_like(joint_bias))
        if self.use_dir_bias:
            if updown is None:
                raise ValueError("model built with use_dir_bias=True but batch has no updown")
            # [B,J,J,H] -> [B,H,J,J], added ON TOP of the -clip(geo,8) scalar (zero-init tables
            # => exact baseline at step 0). Padded pairs stay dominated by PAD_BIAS -1e4.
            dir_b = (self.e_up(updown[..., 0]) + self.e_down(updown[..., 1])).permute(0, 3, 1, 2)
            base = joint_bias.unsqueeze(1) if (joint_bias is not None and joint_bias.dim() == 3) \
                else joint_bias
            joint_bias = dir_b.to(h.dtype) if base is None else base.to(h.dtype) + dir_b.to(h.dtype)

        c = self.t_mlp(timestep_embedding(t * 1000.0, self.dim))
        if text is not None:
            c = c + self.text_mlp(text)
        if blueprint is not None:
            c = c + self.bp_mlp(blueprint)

        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                # use_reentrant=False: the reentrant variant does not play well with DDP's
                # bucketed backward and silently drops the grads of unused parameters.
                h = torch.utils.checkpoint.checkpoint(
                    blk, h, c, joint_bias, frame_valid, joint_valid, rope_cos, rope_sin,
                    trope_cos, trope_sin, use_reentrant=False)
            else:
                h = blk(h, c, joint_bias=joint_bias, frame_valid=frame_valid,
                        joint_valid=joint_valid, rope_cos=rope_cos, rope_sin=rope_sin,
                        trope_cos=trope_cos, trope_sin=trope_sin)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = modulate(self.n_out(h), shift[:, None, None], scale[:, None, None])
        return self.out(h)


class TwoStageInContextDiT(nn.Module):
    """Variant D -- the Kimodo/UMO two-stage denoiser, adapted to per-joint tokens (KTJD-17).

    Contract, verbatim from the sources (kimodo twostage_denoiser.py:36-153, UMO
    kimodo_context_flow_dit.py:1022-1081 -- extracted 2026-08-20):
      - the ROOT tower consumes the FULL noisy state (all joints; Kimodo: root_input_dim =
        input_dim) and emits ONLY the global-root block -- here the root ROW [B,T,1,C];
      - the BODY tower is conditioned on the root tower's OWN x0-prediction from the SAME
        forward pass, DETACHED in training (never GT root; gradients flow at eval "for
        guidance", twostage_denoiser.py:121-130);
      - ONE two-stage forward per ODE step: no root-first-then-body schedule (kimodo_model.py:
        617-633);
      - ONE joint loss on the concatenated output; tower separation is implicit -- root groups
        reach the root tower through the root row, the body loss cannot cross the detached
        bridge (UMO train_hy273_raw_flow.py:1344-1444);
      - towers share NO transformer weights (Kimodo convention: independent embed_text /
        embed_timestep per tower; we inherit that since each InContextMotionDiT owns its own).
    THE DECOMPOSITION (codex round-2 blocker 3: an additive side-channel, or even a raw-root
    replacement, is NOT the cited architecture). Kimodo predicts ONLY the global-root block,
    converts it DETERMINISTICALLY into a local-root representation, removes the global root from
    the body tower's input and puts the local one there instead:

      global root (Kimodo, 5) = smooth_root_pos(3) + heading(2)
      local  root (Kimodo, 4) = heading angular velocity, planar velocity x/z, root height

    KTJD carries the same five quantities in different slots -- smooth-root is XZ only (ch13:15),
    heading is ch15:17, and root HEIGHT lives in the root row's q_position Y (ch1), because KTJD
    subtracts only XZ when forming q_position. So GLOBAL_ROOT_CH = (1, 13, 14, 15, 16), five
    channels, exactly Kimodo's five. Everything else -- all non-root rows, and the root row's
    ch0/ch2 (the XZ residual), rot6d, velocity, contact -- is the body block, matching Kimodo
    (which places every joint's position, the root's included, in the body block).

    Consequently:
      * the root tower's prediction is READ only on GLOBAL_ROOT_CH; its other outputs are unused;
      * the body tower's input has those five slots ZEROED on target frames (the global root is
        REMOVED, as in Kimodo) and receives the local-root block through a zero-init injection --
        so there is no path from the noisy global root into the body tower;
      * the bridge is parameter-free and detached in training (gradients flow at eval, Kimodo's
        allow-guidance convention).

    Demo frames keep their clean global root: they are the in-context reference the whole method
    rests on, and zeroing them would destroy it. Kimodo has no demo, so the question is ours.

    DELIBERATE DEVIATION: Kimodo un-normalizes, differentiates at fps, then re-normalizes with
    dedicated local-root statistics. We have no such statistics, and both the fps factor and the
    re-normalization are constants that the zero-initialised injection absorbs by learning, so
    the bridge here emits raw per-frame differences in normalized units. The QUANTITIES are
    Kimodo's; only their scale convention differs.

    The BODY tower is constructed FIRST: under one seed its backbone WEIGHT init is identical to
    the single-tower arms (the same causal-pairing discipline as the graph-v2 modules); the root
    tower's parameter draws come after.
    """

    GLOBAL_ROOT_CH = (1, 13, 14, 15, 16)      # KTJD analogue of Kimodo's 5-dim global root

    ROOT_TOWER_SEED = 20260820   # see __init__

    def __init__(self, in_ch=17, dim=384, depth=7, n_heads=8, root_dim=192, root_depth=4, **kw):
        super().__init__()
        # Body FIRST, from the ambient RNG stream, so its backbone weights are bit-identical to a
        # single-tower arm under the same seed (the causal-pairing discipline).
        self.body = InContextMotionDiT(in_ch=in_ch, dim=dim, depth=depth, n_heads=n_heads,
                                       local_root_dim=4, **kw)
        # Root SECOND, inside a FORKED stream with a fixed seed (codex round-S5 BLOCK): otherwise
        # any optional module inside the body -- e.g. use_ref_text's projection -- consumes RNG and
        # SHIFTS the root tower's entire initialisation. Measured: 31 changed root tensors and
        # 0.24-0.37 output drift from a module that is itself zero-initialised. Forking makes the
        # root's init depend on nothing but this constant, so body-side options can never move it.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.ROOT_TOWER_SEED)
            self.root = InContextMotionDiT(in_ch=in_ch, dim=root_dim, depth=root_depth,
                                           n_heads=n_heads, **kw)

    @staticmethod
    def _global_to_local(g):
        """[B,T,C] predicted root row -> [B,T,4] local root, Kimodo's parameter-free conversion.
        Angular velocity comes from the signed angle between consecutive heading vectors via
        atan2(cross, dot) -- scale-invariant, so it is well defined even though a PREDICTED
        heading is not unit-norm. The last frame copies the previous pair, as Kimodo does."""
        xz = g[..., 13:15]
        vel = torch.zeros_like(xz)
        vel[:, :-1] = xz[:, 1:] - xz[:, :-1]
        vel[:, -1] = vel[:, -2] if g.shape[1] > 1 else 0.0
        c0, s0 = g[:, :-1, 15], g[:, :-1, 16]
        c1, s1 = g[:, 1:, 15], g[:, 1:, 16]
        om = torch.zeros_like(g[..., 0])
        om[:, :-1] = torch.atan2(c0 * s1 - s0 * c1, c0 * c1 + s0 * s1)
        if g.shape[1] > 1:
            om[:, -1] = om[:, -2]
        return torch.stack([om, vel[..., 0], vel[..., 1], g[..., 1]], dim=-1)

    def forward(self, x, t, *, is_target, **cond):
        root_out = self.root(x, t, is_target=is_target, **cond)
        bridge = root_out[:, :, 0]                                       # [B,T,C], root row
        bridge = bridge.detach() if self.training else bridge            # Kimodo contract
        local = self._global_to_local(bridge)                            # [B,T,4], no parameters

        x_body = x.clone()
        keep = (~is_target)[..., None].to(x.dtype)                       # demo frames stay clean
        for c in self.GLOBAL_ROOT_CH:
            x_body[:, :, 0, c] = x[:, :, 0, c] * keep[..., 0]            # target -> exact zero
        body = self.body(x_body, t, is_target=is_target, local_root=local, **cond)

        out = body.clone()
        for c in self.GLOBAL_ROOT_CH:
            out[:, :, 0, c] = root_out[:, :, 0, c]
        return out


# Semantic loss groups over the AnyTop 13-channel layout, with the ROOT ROW SPLIT OUT.
#
# WHY THE SPLIT (measured, TrueBones 400 clips / 1.96M joint-frames, normalised space):
# a single MSE over [T,J,C] weights each cell by its energy share, which hands the root row only
#   root  3.64% total  ->  ric_pos 0.94% | rot6d 2.28% (heading) | vel 0.42% (global translation)
#   body 96.36% total  ->  ric_pos 24.13% | rot6d 49.32% | vel 21.73% | contact 1.19%
# The root row carries global translation and heading -- exactly the quantities reported as failing
# in generation -- yet it is 1 of J joints (J median 42) so it is diluted to a few percent.
#
# Kimodo (NVIDIA, 2026) computes a SEPARATE loss per representation component and weights those,
# so its root terms are never diluted by joint count; that is what makes its gamma_root = 10.0
# meaningful. Copying its gammas onto an undivided MSE would do nothing, because the dilution
# happens before the weight is applied. So we mirror the structure, not just the numbers:
# each group is averaged over its own elements first, then scaled by gamma.
#
# gammas follow Kimodo Eq.1 (gamma1=gamma3=gamma5=10.0, gamma2=2.0, gamma4=3.0, gamma6=4.0). They
# are a STARTING POINT, not settled: Kimodo trains one human skeleton on 700h of mocap, we train 64
# heterogeneous skeletons on ~1k clips. Kimodo's FK-consistency term (gamma7=5.0) was deliberately
# not included in the first version (run-1 / run-3, to save one differentiable-FK pass per step).
# HISTORY: the 1000-epoch run-1 diagnosis then showed exactly the failure this term exists to
# prevent -- H4 FK<->RIC inconsistency at 4-19% on unseen rigs, monotonically worsening, untouched
# by more training -- and on 2026-08-19 the user directed full Eq.1 alignment ("kimodo有的你都得
# 加"). gamma7 now ships via cfm_loss(gamma_fk=..., fk_pack=...) + src/models/v2/fk_torch.py. The
# launch weight is 0.25, NOT the paper's 5.0: our residual is divided by 0.15 x mean bone length
# (~0.05m on a human), so 0.25 == the SAME physical-unit slope Kimodo's 5.0 applies (5.0 x 0.15 x
# 0.33m), and the measured grad share on real predictions is 0.14-0.35 of the grouped loss
# (calibrated on run-1 ep1000, codex thread 01a01939). The trainer default stays 0.0 so
# pre-alignment checkpoints resume bit-identically.
#
# gamma_root_rot IS THE FIRST ABLATION AXIS, and the one gamma we have positive reason to distrust.
# Measured shares on 300 real TrueBones clips (err2 = x^2, the initial gradient landscape):
#   undivided MSE (today)      root 3.64%   [heading 2.28%]
#   grouped means, no gamma    root 52.51%  [heading 31.22%]
#   grouped means + gammas     root 43.88%  [heading  9.95%]
# So the dilution fix does the heavy lifting; the gammas then CUT heading back by two thirds.
# Kimodo can afford gamma2=2.0 because it conditions heading explicitly on c_dir and augments the
# first-frame heading randomly (its Sec. 4.2/4.3) -- heading is controlled, not learned from the
# loss. We have no c_dir, and wrong turn direction is the reported failure. Sweep {2.0, 5.0, 10.0}
# before trusting 2.0.
KIMODO_GAMMAS = {
    "root_pos": 10.0,   # root row, ch 0:3 and 9:12  (Kimodo r^p)
    "root_rot": 2.0,    # root row, ch 3:9           (Kimodo r^a)
    "body_pos": 10.0,   # non-root,  ch 0:3          (Kimodo j^p)
    "body_rot": 10.0,   # non-root,  ch 3:9          (Kimodo j^a)
    "body_vel": 3.0,    # non-root,  ch 9:12         (Kimodo j^v)
    "contact": 4.0,     # all rows,  ch 12           (Kimodo f)
}

# (row-slice, channel-index-list) per group. Row slice is applied on the joint axis.
_GROUP_SPEC = {
    "root_pos": (slice(0, 1), [0, 1, 2, 9, 10, 11]),
    "root_rot": (slice(0, 1), [3, 4, 5, 6, 7, 8]),
    "body_pos": (slice(1, None), [0, 1, 2]),
    "body_rot": (slice(1, None), [3, 4, 5, 6, 7, 8]),
    "body_vel": (slice(1, None), [9, 10, 11]),
    "contact": (slice(0, None), [12]),
}

# KTJD-17 layout (README:47-57): ch0:3 canonical positions (XZ minus smooth-root), 3:9 global
# delta rot6d, 9:12 velocities (SUPERVISION-ONLY; decode never integrates them), 12 contact,
# root-only 13:15 smooth-root XZ, 15:17 heading. Root position and root velocity are SEPARATE
# groups (codex 01a01b1a round-KTJD: their semantics and gains differ), and the two new root
# blocks get their own groups. STARTING gammas mirror the Kimodo mapping; they MUST be
# re-measured on KTJD-normalized energies before any real run (the 13ch measurement does not
# transfer: positions changed frame, rotations changed convention).
_GROUP_SPEC_KTJD17 = {
    "root_pos": (slice(0, 1), [0, 1, 2]),
    "root_rot": (slice(0, 1), [3, 4, 5, 6, 7, 8]),
    "root_vel": (slice(0, 1), [9, 10, 11]),
    "body_pos": (slice(1, None), [0, 1, 2]),
    "body_rot": (slice(1, None), [3, 4, 5, 6, 7, 8]),
    "body_vel": (slice(1, None), [9, 10, 11]),
    "contact": (slice(0, None), [12]),
    "smooth_root": (slice(0, 1), [13, 14]),
    "heading": (slice(0, 1), [15, 16]),
}
# MEASURED on 300 real KTJD-17 train clips in normalized space (scratch/_measure_ktjd17_energy.py,
# 2026-08-20; per-element energies E_i): root_pos .484 root_rot .333 root_vel .846 body_pos .868
# body_rot .333 body_vel 1.329 contact .141 smooth_root 2.781 heading .500. Under share ~ gamma^2*E
# the naive Kimodo-mapped gammas hand smooth_root 58.9% of the gradient (its normalized energy is
# ~2.8, far above the calibration's unit target). Gammas below are solved from a PRE-REGISTERED
# target share profile -- body_pos 21% / body_rot 22% / smooth_root 23% (trajectory prominent, not
# dominant; the floating failure lives here) / root_pos 12% / heading 5% (wrong-turn-direction was
# the reported 13ch failure axis; keep it alive) / body_vel 5.5% / contact 4.6% / root_rot 3.6% /
# root_vel 3.5% -- via gamma_i = sqrt(share_i/E_i), scaled to anchor body_rot at 10.0.
# REFERENCE ONLY -- never consumed by training (codex round-S0: placeholders must refuse, not
# train). The trainer loads gammas exclusively from the versioned calibration artifact
# (configs/ktjd17_gamma_calibration_v1.json, built by scripts/_measure_ktjd17_gamma_calibration.py
# AFTER the crop-origin re-base fix; the 2026-08-20 pre-rebase measurement above is stale for
# smooth_root, whose energy included clip-level origin offsets).
KTJD17_GAMMAS_PREREBASE_REFERENCE = {
    "root_pos": 6.0, "root_rot": 4.0, "root_vel": 2.5,
    "body_pos": 6.0, "body_rot": 10.0, "body_vel": 2.5,
    "contact": 7.0, "smooth_root": 3.5, "heading": 4.0,
}

# Bumped whenever the KTJD validity treatment changes semantics. v2 = "effective validity"
# applied to x1/noise/anchor/interpolant/supervision in cfm_loss AND per-step state projection
# in sample() (codex round-S0 complete treatment); v1 was noise+loss masking only.
KTJD17_MASK_POLICY = "ktjd17_effvalid_v2"


def grouped_loss(err2, m, gammas, group_spec=None):
    """err2 [B,T,J,C] squared error, m [B,T,J,C] validity mask, -> (total, per-group dict).

    SIZE-INVARIANT GRADIENT SHARES. A plain per-group mean does NOT equalise groups -- it
    over-corrects, because a small group's mean gives each of its elements a larger derivative:

        loss_i  = gamma_i * (1/N_i) * sum_j err2_ij
        d loss / d err_ij            = 2 * gamma_i * err_ij / N_i
        sum of squared grads in i    = N_i * (2 gamma_i E_i / N_i)^2  ~  gamma_i^2 * E_i / N_i

    so the share goes UP as the group shrinks. Measured on a real batch (mean J 53.8): plain
    per-group means handed the root row 95.14% of |dLoss/dx|^2 -- N_body/N_root ~ 26.5 predicts
    26.5/27.5 = 96.4%, which is what happened. That starves the body, and (measured in the same
    run) left the demo path unused: with 95% of the gradient on the root row, the model fits global
    translation and heading, which the TEXT already predicts, so the demo has nothing to contribute.

    Multiplying each group by sqrt(N_i / N_total) cancels the 1/N_i exactly:

        share_i  ~  (gamma_i^2 * N_i/N_tot) * E_i / N_i  =  gamma_i^2 * E_i / N_tot

    i.e. a group's gradient share depends only on its gamma and its signal energy, never on how many
    joints or channels it spans. That is the property the design wanted and the plain mean did not
    deliver. Predicted shares under this form (using measured per-element energies): root 40.5%.
    """
    spec = _GROUP_SPEC if group_spec is None else group_spec
    total, parts = 0.0, {}
    counts = {}
    for name in gammas:
        js, cs = spec[name]
        counts[name] = m[:, :, js][..., cs].sum()
    n_total = sum(counts.values()).clamp_min(1.0)
    for name, gamma in gammas.items():
        js, cs = spec[name]
        e = err2[:, :, js][..., cs]
        mm = m[:, :, js][..., cs]
        denom = counts[name].clamp_min(1.0)
        part = (e * mm).sum() / denom
        parts[name] = part.detach()
        total = total + gamma * torch.sqrt(denom / n_total) * part
    return total, parts


def cfm_loss(model, x1, *, is_target, valid=None, gammas=None, return_parts=False,
             t_sampler="uniform", v_space=False, sigma_min=0.05, huber_delta=0.0,
             gamma_fk=0.0, fk_pack=None, group_spec=None, gamma_vel=0.0, gamma_lock=0.0,
             gamma_acc=0.0,
             channel_valid=None, heading_valid=None, anchor=None, **cond):
    """Conditional flow matching with **x-prediction**: the network outputs the clean motion.

    `gammas=None` keeps the original single unweighted MSE (kept as the ablation baseline).
    `gammas=KIMODO_GAMMAS` uses the grouped, root-undiluted objective described above.

    WHY X-PREDICTION, NOT V-PREDICTION (measured on this very prototype):
      With v-prediction the network must output v = x1 - x0 = x1 - x_t at t=0, i.e. it has to learn
      a "-identity" term through LayerNorms and nonlinearities. Measured after full convergence
      (loss 0.009 = 0.55% of variance): cos(v_pred, v_true) was +0.9993 at t=0.5 and +0.9980 at
      t=0.9 but only +0.7060 at t=0.1, with the scale down at 0.70. Sampling starts exactly at t=0,
      so that weak corner poisons the whole trajectory and relL2 stayed at 1.18-1.38 no matter how
      many ODE steps were used. Predicting x1 removes the problem: the target is independent of x0.
      This also matches the v1 ablation, where x-prediction beat v-prediction by ~2.2x on geometry.
    """
    B, T, J, C = x1.shape
    if t_sampler == "logitnormal":
        # ACMDM-JiT / SD3 concentrate sampling toward the DATA end. Their convention puts clean
        # data at t=0, ours at t=1, so their mu=-0.8 flips sign here: sigmoid(+0.8) centres the
        # draw at t~0.69, toward OUR clean end. The verbatim -0.8 form (which early v2 runs did
        # use, before the v-space weight existed) was refused at review for the CURRENT weighted
        # objective (codex 2026-08-28): under the v-space weight it moved clean-end (t>=0.8)
        # supervision mass from 89.74% to 4.36% -- the opposite of the sampler's purpose. Measured
        # composition with sigma_min=0.05: p(t)*w(t) mass at t>=0.8 is 69.9%, and the extreme
        # endpoint band t>=0.95 drops from uniform's 51.3% to 6.3% -- supervision moves from the
        # noisy-gradient endpoint into the information-rich near-clean band, which is the SD3
        # rationale for the sampler.
        t = torch.sigmoid(torch.randn(B, device=x1.device) * 0.8 + 0.8)
    else:
        t = torch.rand(B, device=x1.device)
    # ---- EFFECTIVE VALIDITY (KTJD-17, codex round-S0 "complete treatment"): one mask, applied
    # to EVERY tensor that carries state -- clean x1 (demo frames included), base noise, anchor,
    # hence the interpolant and the supervision. A loss-only or noise-only mask is not enough:
    # fixed_dof rows keep nonzero inherited d6 in raw GT, and heading-invalid frames carry
    # unreliable heading values; both must be canonical ZERO in model space at every t.
    eff = None
    if channel_valid is not None:
        eff = channel_valid[:, None].to(x1.dtype).expand(B, T, J, C).clone()
        if heading_valid is not None and C >= 17:
            # [B,T] flag gates the ROOT heading channels per frame (root is row 0 by contract)
            eff[:, :, 0, 15:17] *= heading_valid.to(x1.dtype)[:, :, None]
        x1 = x1 * eff
    x0 = torch.randn_like(x1)
    if anchor is not None:
        # UMO source-centered base (cfgA:70 aligned_source_plus_standard_gaussian_v1): the flow
        # base becomes anchor + N(0,I), so "copy the anchor" is the zero-work default and the
        # conditioning only has to steer the DEVIATION. The anchor is BASE GEOMETRY, not a
        # condition: it is identical across all CFG branches and is never dropped.
        x0 = x0 + anchor.to(x0.dtype)
    if eff is not None:
        # project the SUM (noise + anchor): invalid cells are exact zero in the base too, so
        # xt == x1 == 0 there at every t and nothing leaks through attention.
        x0 = x0 * eff
    tt = t[:, None, None, None]
    xt = (1 - tt) * x0 + tt * x1
    xt = torch.where((~is_target)[..., None, None], x1, xt)   # demo frames stay clean (projected)
    x1_pred = model(xt, t, is_target=is_target, **cond)
    # `valid` is REQUIRED, not optional. Without it, padded joints (a J=9 rig in a batch whose max
    # is 142 is 94% padding) count as legitimate zero targets and dominate every group denominator.
    if valid is None:
        raise ValueError("cfm_loss requires `valid` [B,T,J] = frame_valid & joint_valid; passing "
                         "None silently trains on padding as valid zeros")
    m = (is_target[..., None] & valid).to(x1.dtype)[..., None]
    m = m.expand_as(x1)
    # REAL target frames only (codex round-S2 #3): `valid` = frame_valid & joint_valid, so
    # valid.any(-1) is the per-frame real-frame flag. The physical-space terms below take
    # DIFFERENCES between consecutive frames, and a padded frame is an exact zero -- which in the
    # rest-centered space IS the rest pose. Feeding padding to them manufactures a
    # last-real-frame -> rest jump that no motion contains, and poisons the speed diagnostic.
    real_target = is_target & valid.any(-1)
    if eff is not None:
        m = m * eff
    if huber_delta > 0.0:
        # ROBUST TARGET (user-agreed 2026-08-21). The corpus carries source-side defects that no
        # detector separated from fast motion across seven attempts -- isolated teleports, sustained
        # oscillation, opening IK settle, and quaternion-induced errors already present in the
        # source BVH (Cobra MANIS issue 381). Rather than keep hunting for a criterion, the
        # objective is made robust so undetected contamination cannot dominate: below the knee the
        # term is EXACTLY the squared error, so the calibrated gammas and their group shares are
        # untouched for the overwhelming majority of cells; above it the term grows linearly and
        # the per-cell gradient saturates at 2*delta instead of growing with the error.
        # delta=10 sits just under the measured p99.99 of |normalized target| (11.45 over 4.63e8
        # supervised cells): 99.982% of supervision is unaffected, while the worst cell in the
        # corpus (bound 62.7) has its gradient cut ~6x. As training converges the errors shrink
        # and the knee stops binding, so this is not a permanent reweighting of the objective.
        d = (x1_pred - x1).abs()
        err2 = torch.where(d <= huber_delta, d ** 2, huber_delta * (2.0 * d - huber_delta))
    else:
        err2 = (x1_pred - x1) ** 2
    if v_space:
        # JiT (denoiser.py:58-62): the network predicts clean data, but the LOSS lives in velocity
        # space. On the OT path x1 - xt = (1-t)(x1 - x0), so v-space MSE == x-space MSE weighted by
        # 1/(1-t)^2 -- emphasis lands on the near-data regime where high-frequency detail (jitter)
        # is decided. The clamp (user's ACMDM-JiT sigma_min=0.05) caps the weight at 400x.
        w = 1.0 / torch.clamp(1.0 - t, min=sigma_min) ** 2
        # UNIT-MEAN NORMALIZATION (2026-08-20, deliberate deviation from UMO's literal form).
        # E_t~U(0,1)[1/max(1-t,s)^2] = 2/s - 1 (= 39 at s=0.05); the raw weight otherwise
        # inflates the loss and gradients by that factor (measured smoke: grad mean 249 -> 12226,
        # every step clipped ~1e4x). Dividing by the SAMPLER'S OWN mean keeps the RELATIVE
        # up-weighting of clean-end timesteps while restoring scale, so the flow term keeps its
        # calibrated balance against the un-weighted FK/velocity/lock terms. The constant MUST
        # match the active sampler (codex 2026-08-28): using uniform's 39 under logit-normal
        # would shrink the flow term to 0.61x and silently re-weight the auxiliaries by 1.64x.
        if t_sampler == "logitnormal":
            # E[1/max(1-t,0.05)^2] over t = sigmoid(0.8 N + 0.8): Gauss quadrature, abserr 1e-7.
            # BOUND to the sampler's (mu=+0.8, sigma=0.8) and sigma_min=0.05 hard-coded above --
            # recompute if any of the three changes.
            assert abs(sigma_min - 0.05) < 1e-12, \
                "logitnormal unit-mean constant is precomputed for sigma_min=0.05 only"
            w_mean = 23.733256
        else:
            w_mean = 2.0 / sigma_min - 1.0
        err2 = err2 * (w / w_mean)[:, None, None, None]
    if gammas is None:
        loss, parts = (err2 * m).sum() / m.sum().clamp_min(1.0), {}
    else:
        loss, parts = grouped_loss(err2, m, gammas, group_spec=group_spec)
    if gamma_fk > 0.0:
        # Kimodo Eq.1 term 7 (gamma7=5.0): FK(pred rotations) vs RIC(pred positions) consistency.
        # Added 2026-08-19 (user: "kimodo有的你都得加") after the 1000-epoch run-1 verdict that the
        # FK<->RIC split (H4, 4-19% on unseen rigs) does not train away without being penalized.
        # The caller passes gamma_fk ALREADY warmup-ramped (hy273 recipe: linear over
        # fk_warmup_steps); NOT weighted by the v_space 1/(1-t)^2 factor -- consistency is a
        # property of x1_pred at every t, and both references apply it unweighted.
        if fk_pack is None:
            raise ValueError("gamma_fk > 0 requires fk_pack (mean/std/parents/offsets/n_joints); "
                             "build the dataset with emit_fk_fields=True")
        if fk_pack.get("kind") == "ktjd17":
            # KTJD-17 gamma7: FK vs DIRECT positions via the official decoder semantics
            # (kimodo Eq.1 term 7, KTJD form -- user 2026-08-20: kimodo-like loss aligned)
            from src.models.v2.fk_torch import fk_ktjd_consistency_loss
            fk_term, fk_dist = fk_ktjd_consistency_loss(
                x1_pred, fk_pack["anytop_mean"], fk_pack["anytop_std"], fk_pack["std_floor"],
                fk_pack["parents"], fk_pack["rest_offsets"], fk_pack["R_rest_global"],
                fk_pack["n_joints"], frame_mask=real_target, want_diag=return_parts)
        else:
            from src.models.graph_salad.world_recovery import recover_world_positions_torch
            from src.models.v2.fk_torch import fk_ric_consistency_loss
            fk_term, fk_dist = fk_ric_consistency_loss(
                x1_pred, fk_pack["anytop_mean"], fk_pack["anytop_std"], fk_pack["std_floor"],
                fk_pack["parents"], fk_pack["rest_offsets"], fk_pack["n_joints"],
                frame_mask=real_target, ric_world_fn=recover_world_positions_torch,
                want_diag=return_parts)
        loss = loss + gamma_fk * fk_term
        if return_parts:
            # scalar conversions sync; keep them off the per-step train path (return_parts=False)
            parts = dict(parts)
            parts["fk_consist"] = float(fk_term.detach())
            parts["fk_dist"] = fk_dist   # mean |FK-RIC| in bone-length units, weight-0 diagnostic
    if gamma_vel > 0.0 or gamma_lock > 0.0:
        # UMO's anti-degenerate pair (train_hy273_raw_flow.py:733-757, weights 0.01 each):
        # physical-space finite differences a frozen output cannot satisfy, plus the contact lock
        # that stops the opposite failure. Independent of gamma_fk (different job: gamma_fk makes
        # the two DECODES agree -- a static pose satisfies it perfectly -- while these two make the
        # motion MOVE). KTJD-only: it needs the direct-decode geometry, so it rides fk_pack.
        if fk_pack is None or fk_pack.get("kind") != "ktjd17":
            raise ValueError("gamma_vel/gamma_lock need the KTJD fk_pack (kind='ktjd17')")
        from src.models.v2.fk_torch import ktjd_dynamics_losses
        vel_term, lock_term, speed_ratio, speed_n = ktjd_dynamics_losses(
            x1_pred, x1, fk_pack["anytop_mean"], fk_pack["anytop_std"], fk_pack["std_floor"],
            fk_pack["rest_offsets"], fk_pack["n_joints"], frame_mask=real_target,
            # contact must be thresholded on the DE-NORMALIZED channel (codex round-S7 blocker
            # 3): per-cell standardization moves the 0/1 flag off {0,1}, and testing >0.5 on the
            # normalized value missed ~19% of true contact events, constant-contact cells included.
            contact_on=((x1[..., 12] * (fk_pack["anytop_std"][:, None, :, 12] + fk_pack["std_floor"])
                         + fk_pack["anytop_mean"][:, None, :, 12]) > 0.5)
            if gamma_lock > 0.0 else None,
            want_diag=return_parts,
            lock_denominator=fk_pack.get("lock_denominator"))   # augmented samples: the pre-pruning pair count (fk_torch)
        loss = loss + gamma_vel * vel_term + gamma_lock * lock_term
        if return_parts:
            parts = dict(parts)
            parts["dyn_vel"] = float(vel_term.detach())
            parts["foot_lock"] = float(lock_term.detach())
            # speed_ratio = predicted mean joint speed / GT's. 1.0 matched, ~0 frozen: the ONLINE
            # frozen-pose monitor, so the ep250 surprise cannot repeat unseen.
            parts["speed_ratio"] = speed_ratio
            parts["speed_ratio_n"] = speed_n      # windows that actually contributed a ratio
    if gamma_acc > 0.0:
        # ACCELERATION MATCHING (user plan-a, 2026-08-28). Measured attribution
        # (jitter_analysis_s32.txt): the residual micro-jitter is uniform >5Hz noise
        # concentrated on near-static joints (gen/GT band ratio 2.9x at 0-2Hz rising to 32.8x at
        # 10-15Hz; GT's own high-band share 0.57%; FK/RIC parity 1.00), i.e. per-frame
        # independent error in the x1 prediction itself, which neither the JiT weighting nor
        # more ODE steps nor a one-euro post-filter resolved to the user's eye. This term matches
        # the SECOND DIFFERENCE of the prediction to GT's on the normalized channels: for
        # near-static joints GT's acceleration is ~0 so frame noise is pushed to zero, while
        # moving joints' real bursts are the TARGET, not a casualty (unlike a smoothness prior).
        # Deliberately NOT v-space weighted -- like gamma_fk/vel/lock it is a geometric
        # consistency term. GT d2-energy is ~10% of position energy (measured over 160 batches),
        # so gamma_acc=1.0 puts this near a 9% share of the flow term at init.
        d2p = x1_pred[:, 2:] - 2 * x1_pred[:, 1:-1] + x1_pred[:, :-2]
        d2t = x1[:, 2:] - 2 * x1[:, 1:-1] + x1[:, :-2]
        # a triple of frames supervises acceleration only when ALL THREE are real target frames
        m3 = m[:, 2:] * m[:, 1:-1] * m[:, :-2]
        acc_term = ((d2p - d2t) ** 2 * m3).sum() / m3.sum().clamp_min(1.0)
        loss = loss + gamma_acc * acc_term
        if return_parts:
            parts = dict(parts)
            parts["acc_match"] = float(acc_term.detach())
    return (loss, parts) if return_parts else loss


@torch.no_grad()
def sample(model, x_ref, is_target, steps, cfg_text=1.0, cfg_demo=1.0, demo_frames=None,
           channel_valid=None, anchor=None, heading_valid=None, **cond):
    """Euler along the straight path, driven by the predicted clean motion.

    The update x <- x + (x1_pred - x)/(steps - i) walks the remaining distance in the remaining
    steps. It is numerically stable at t->1 (the denominator is a step count, never 1-t -> 0),
    which is precisely where the v1 x-prediction sampler blew up by 25x.
    channel_valid [B,J,C]: KTJD-17 invalid cells stay EXACT ZERO through the whole ODE -- the
    state is re-projected after EVERY Euler update (codex round-S0: init-only zeroing lets the
    model's own prediction resurrect invalid cells on step one), mirroring training.
    heading_valid [B,T] is DEMO-side only: the demo's flag is a legitimate input at inference,
    but target-time validity is unknowable at generation and is never consumed here -- target
    heading is generated, and any validity call happens downstream from decoded rotations.
    """
    cv = channel_valid[:, None].to(x_ref.dtype) if channel_valid is not None else None
    if cv is not None:
        x_ref = x_ref * cv
        if heading_valid is not None and x_ref.shape[-1] >= 17:
            hv = heading_valid.to(x_ref.dtype).clone()
            hv[is_target] = 1.0                    # never gate target frames with GT validity
            x_ref[:, :, 0, 15:17] *= hv[:, :, None]
    noise = torch.randn_like(x_ref)
    if anchor is not None:
        noise = noise + anchor.to(noise.dtype)     # base = anchor + N(0,I), matching training
    if cv is not None:
        noise = noise * cv                          # project the SUM, as training does
    x = torch.where(is_target[..., None, None], noise, x_ref)

    guided = (cfg_text != 1.0) or (cfg_demo != 1.0)
    if guided:
        # Compositional dual guidance on the x1-prediction (InstructPix2Pix-style):
        #   x_hat = x_uu + cfg_demo * (x_du - x_uu) + cfg_text * (x_dt - x_du)
        # where u/d mark dropped/kept demo and text. Requires a CFG-trained model (independent
        # demo/text dropout); both scales at 1.0 reduce to a single forward, bit-identical to the
        # unguided path.
        cond_dt = cond
        # Uncond text must be the ZERO VECTOR, exactly as training's dropout produces it: with
        # text=None the model skips the text_mlp branch entirely, a c the network never saw.
        cond_du = ({**cond, "text": torch.zeros_like(cond["text"])}
                   if cond.get("text") is not None else cond)
        if cfg_demo != 1.0:
            # Only the demo-guided path needs the demo-drop machinery (codex 2026-08-27 r2):
            # text-only guidance must neither require demo_frames nor build cond_uu.
            assert demo_frames is not None, \
                "guided sampling needs demo_frames to build the demo-drop"
            fv = cond.get("frame_valid")
            fv_u = fv.clone() if fv is not None else None
            if fv_u is not None:
                fv_u[:, :demo_frames] = False
            cond_uu = {**cond_du, "frame_valid": fv_u}
            if cond.get("demo_text") is not None:
                # the demo's caption IS part of the demo condition: the demo-dropped branch must
                # not keep a description of frames it can no longer see (mirrors apply_cfg_drops).
                cond_uu = {**cond_uu, "demo_text": torch.zeros_like(cond["demo_text"])}

    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        if not guided:
            x1_pred = model(x, t, is_target=is_target, **cond)
        elif cfg_demo == 1.0:
            # Text-only guidance. x_uu's algebraic coefficient is (1 - cfg_demo) = 0, but
            # evaluating it anyway leaks the UNTRAINED demo-drop branch into the state at
            # floating-point rounding level on every Euler step (codex 2026-08-27) -- and burns
            # a third forward for nothing. Both surviving branches are p_drop_text-trained.
            x_dt = model(x, t, is_target=is_target, **cond_dt)
            x_du = model(x, t, is_target=is_target, **cond_du)
            x1_pred = x_du + cfg_text * (x_dt - x_du)
        else:
            x_dt = model(x, t, is_target=is_target, **cond_dt)
            x_du = model(x, t, is_target=is_target, **cond_du)
            x_u = x.clone(); x_u[:, :demo_frames] = 0.0
            x_uu = model(x_u, t, is_target=is_target, **cond_uu)
            x1_pred = x_uu + cfg_demo * (x_du - x_uu) + cfg_text * (x_dt - x_du)
        step = (x1_pred - x) / (steps - i)
        x = torch.where(is_target[..., None, None], x + step, x_ref)
        if cv is not None:
            x = x * cv                              # per-step state projection (see docstring)
    return x
