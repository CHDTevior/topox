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
import torch.nn.functional as F


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
    skeleton graph on the joint axis) and an optional key-padding mask."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        assert dim % n_heads == 0
        self.h, self.dh = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, attn_bias=None, key_pad=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        bias = None
        if attn_bias is not None:
            bias = attn_bias if attn_bias.dim() == 4 else attn_bias.unsqueeze(1)
            bias = bias.expand(B, self.h, N, N).clone()
        if key_pad is not None:                       # [B,N] True = valid
            m = torch.zeros(B, 1, 1, N, device=x.device, dtype=q.dtype)
            m = m.masked_fill(~key_pad[:, None, None, :], torch.finfo(q.dtype).min)
            bias = m if bias is None else bias + m
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    """One factorised block: temporal attention -> spatial (joint) attention -> MLP,
    every sub-layer AdaLN-modulated by the conditioning vector c."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1, self.n2, self.n3 = (nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6) for _ in range(3))
        self.t_attn = Attention(dim, n_heads)
        self.s_attn = Attention(dim, n_heads)
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

    def forward(self, x, c, joint_bias=None, frame_valid=None, joint_valid=None):
        # x [B,T,J,D]
        B, T, J, D = x.shape
        p = self.ada(c)                                    # [B, 9D]
        (st, sc, sg, st2, sc2, sg2, mt, mc, mg) = p.chunk(9, dim=-1)
        e = lambda v: v[:, None, None, :]                  # broadcast over T,J

        # --- temporal: tokens are frames, batched over joints ---
        y = modulate(self.n1(x), e(st), e(sc)).permute(0, 2, 1, 3).reshape(B * J, T, D)
        fv = frame_valid.repeat_interleave(J, 0) if frame_valid is not None else None
        y = self.t_attn(y, key_pad=fv).reshape(B, J, T, D).permute(0, 2, 1, 3)
        x = x + e(sg) * y

        # --- spatial: tokens are joints, batched over frames; skeleton graph enters as bias ---
        y = modulate(self.n2(x), e(st2), e(sc2)).reshape(B * T, J, D)
        jb = joint_bias.repeat_interleave(T, 0) if joint_bias is not None else None
        jv = joint_valid.repeat_interleave(T, 0) if joint_valid is not None else None
        y = self.s_attn(y, attn_bias=jb, key_pad=jv).reshape(B, T, J, D)
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
                 d_joint_sem=4096, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.x_in = nn.Linear(in_ch, dim)
        self.mask_token = nn.Parameter(torch.zeros(dim))          # marks "this frame is to be generated"
        # TEMPORAL POSITION. Without this the model cannot tell frame 1 from frame 16, so when the
        # input is pure noise (t->0) it has no way to know what belongs at each timestep and can only
        # emit the per-frame mean. Measured before adding it: error at t=0 was 0.997 (i.e. nothing
        # learned) while error at t=0.7 was 0.024 (copying the input, which needs no position), and
        # the conditioning effect sat at 0.010 no matter how the conditioning was scaled -- AdaLN
        # broadcasts one [D] shift/scale to every token, so without positions it cannot paint a
        # T x J x C target that differs per position.
        self.max_T, self.max_J = 4096, 160
        self.t_pos = nn.Parameter(torch.zeros(1, self.max_T, 1, dim))
        # JOINT POSITION, separate from joint SEMANTICS. Semantics say what a joint IS
        # ("the left claw of the arm"); position says which slot it occupies. They are complementary:
        # several joints down one limb have near-identical descriptions, and joint_semantics is
        # optional, so without an index embedding the model cannot address individual joints at all.
        # Same failure as the missing temporal position: fine when copying the input, useless when
        # generating from noise.
        self.j_pos = nn.Parameter(torch.zeros(1, 1, self.max_J, dim))
        nn.init.normal_(self.t_pos, std=0.02); nn.init.normal_(self.j_pos, std=0.02)
        self.joint_sem = nn.Linear(d_joint_sem, dim)              # per-joint identity (LLM2Vec)
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.text_mlp = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, dim), nn.SiLU(),
                                      nn.Linear(dim, dim))
        # blueprint is low-dimensional and explicit: it modulates every block and cannot be routed around
        self.bp_mlp = nn.Sequential(nn.Linear(d_blueprint, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([Block(dim, n_heads, mlp_ratio) for _ in range(depth)])
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

    def forward(self, x, t, *, is_target, joint_sem=None, text=None, blueprint=None,
                joint_bias=None, frame_valid=None, joint_valid=None):
        """x [B,T,J,C] ; t [B] in [0,1] ; is_target [B,T] bool."""
        B, T, J, _ = x.shape
        h = self.x_in(x)
        h = h + self.t_pos[:, :T] + self.j_pos[:, :, :J]                        # temporal + joint position
        h = h + is_target[..., None, None].to(h.dtype) * self.mask_token       # flag frames to generate
        if joint_sem is not None:
            h = h + self.joint_sem(joint_sem)[:, None]                          # [B,1,J,D]

        c = self.t_mlp(timestep_embedding(t * 1000.0, self.dim))
        if text is not None:
            c = c + self.text_mlp(text)
        if blueprint is not None:
            c = c + self.bp_mlp(blueprint)

        for blk in self.blocks:
            h = blk(h, c, joint_bias=joint_bias, frame_valid=frame_valid, joint_valid=joint_valid)
        shift, scale = self.ada_out(c).chunk(2, dim=-1)
        h = modulate(self.n_out(h), shift[:, None, None], scale[:, None, None])
        return self.out(h)


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


def grouped_loss(err2, m, gammas):
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
    total, parts = 0.0, {}
    counts = {}
    for name in gammas:
        js, cs = _GROUP_SPEC[name]
        counts[name] = m[:, :, js][..., cs].sum()
    n_total = sum(counts.values()).clamp_min(1.0)
    for name, gamma in gammas.items():
        js, cs = _GROUP_SPEC[name]
        e = err2[:, :, js][..., cs]
        mm = m[:, :, js][..., cs]
        denom = counts[name].clamp_min(1.0)
        part = (e * mm).sum() / denom
        parts[name] = part.detach()
        total = total + gamma * torch.sqrt(denom / n_total) * part
    return total, parts


def cfm_loss(model, x1, *, is_target, valid=None, gammas=None, return_parts=False,
             t_sampler="uniform", v_space=False, sigma_min=0.05,
             gamma_fk=0.0, fk_pack=None, **cond):
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
        # ACMDM-JiT / SD3: logit-normal(mu=-0.8, sigma=0.8) concentrates supervision where the
        # x->v conversion is worst-conditioned, instead of spreading it uniformly.
        t = torch.sigmoid(torch.randn(B, device=x1.device) * 0.8 - 0.8)
    else:
        t = torch.rand(B, device=x1.device)
    x0 = torch.randn_like(x1)
    tt = t[:, None, None, None]
    xt = (1 - tt) * x0 + tt * x1
    xt = torch.where((~is_target)[..., None, None], x1, xt)   # demo frames stay clean
    x1_pred = model(xt, t, is_target=is_target, **cond)
    # `valid` is REQUIRED, not optional. Without it, padded joints (a J=9 rig in a batch whose max
    # is 142 is 94% padding) count as legitimate zero targets and dominate every group denominator.
    if valid is None:
        raise ValueError("cfm_loss requires `valid` [B,T,J] = frame_valid & joint_valid; passing "
                         "None silently trains on padding as valid zeros")
    m = (is_target[..., None] & valid).to(x1.dtype)[..., None]
    m = m.expand_as(x1)
    err2 = (x1_pred - x1) ** 2
    if v_space:
        # JiT (denoiser.py:58-62): the network predicts clean data, but the LOSS lives in velocity
        # space. On the OT path x1 - xt = (1-t)(x1 - x0), so v-space MSE == x-space MSE weighted by
        # 1/(1-t)^2 -- emphasis lands on the near-data regime where high-frequency detail (jitter)
        # is decided. The clamp (user's ACMDM-JiT sigma_min=0.05) caps the weight at 400x.
        w = 1.0 / torch.clamp(1.0 - t, min=sigma_min) ** 2
        err2 = err2 * w[:, None, None, None]
    if gammas is None:
        loss, parts = (err2 * m).sum() / m.sum().clamp_min(1.0), {}
    else:
        loss, parts = grouped_loss(err2, m, gammas)
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
        from src.models.graph_salad.world_recovery import recover_world_positions_torch
        from src.models.v2.fk_torch import fk_ric_consistency_loss
        fk_term, fk_dist = fk_ric_consistency_loss(
            x1_pred, fk_pack["anytop_mean"], fk_pack["anytop_std"], fk_pack["std_floor"],
            fk_pack["parents"], fk_pack["rest_offsets"], fk_pack["n_joints"],
            frame_mask=is_target, ric_world_fn=recover_world_positions_torch,
            want_diag=return_parts)
        loss = loss + gamma_fk * fk_term
        if return_parts:
            # scalar conversions sync; keep them off the per-step train path (return_parts=False)
            parts = dict(parts)
            parts["fk_consist"] = float(fk_term.detach())
            parts["fk_dist"] = fk_dist   # mean |FK-RIC| in bone-length units, weight-0 diagnostic
    return (loss, parts) if return_parts else loss


@torch.no_grad()
def sample(model, x_ref, is_target, steps, cfg_text=1.0, cfg_demo=1.0, demo_frames=None, **cond):
    """Euler along the straight path, driven by the predicted clean motion.

    The update x <- x + (x1_pred - x)/(steps - i) walks the remaining distance in the remaining
    steps. It is numerically stable at t->1 (the denominator is a step count, never 1-t -> 0),
    which is precisely where the v1 x-prediction sampler blew up by 25x.
    """
    x = torch.where(is_target[..., None, None], torch.randn_like(x_ref), x_ref)

    guided = (cfg_text != 1.0) or (cfg_demo != 1.0)
    if guided:
        # Compositional dual guidance on the x1-prediction (InstructPix2Pix-style):
        #   x_hat = x_uu + cfg_demo * (x_du - x_uu) + cfg_text * (x_dt - x_du)
        # where u/d mark dropped/kept demo and text. Requires a CFG-trained model (independent
        # demo/text dropout); both scales at 1.0 reduce to a single forward, bit-identical to the
        # unguided path.
        assert demo_frames is not None, "guided sampling needs demo_frames to build the demo-drop"
        cond_dt = cond
        # Uncond text must be the ZERO VECTOR, exactly as training's dropout produces it: with
        # text=None the model skips the text_mlp branch entirely, a c the network never saw.
        cond_du = ({**cond, "text": torch.zeros_like(cond["text"])}
                   if cond.get("text") is not None else cond)
        fv = cond.get("frame_valid")
        fv_u = fv.clone() if fv is not None else None
        if fv_u is not None:
            fv_u[:, :demo_frames] = False
        cond_uu = {**cond_du, "frame_valid": fv_u}

    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        if not guided:
            x1_pred = model(x, t, is_target=is_target, **cond)
        else:
            x_dt = model(x, t, is_target=is_target, **cond_dt)
            x_du = model(x, t, is_target=is_target, **cond_du)
            x_u = x.clone(); x_u[:, :demo_frames] = 0.0
            x_uu = model(x_u, t, is_target=is_target, **cond_uu)
            x1_pred = x_uu + cfg_demo * (x_du - x_uu) + cfg_text * (x_dt - x_du)
        step = (x1_pred - x) / (steps - i)
        x = torch.where(is_target[..., None, None], x + step, x_ref)
    return x
