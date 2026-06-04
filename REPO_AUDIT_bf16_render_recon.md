Reading additional input from stdin...
OpenAI Codex v0.135.0
--------
workdir: /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: xhigh
reasoning summaries: none
session id: 019e8e67-3cd3-7331-ab28-bf9507330ebc
--------
user
You are an independent repo auditor. Verify these claims against the actual repository. Trust nothing except code and command evidence you inspect yourself.

Scope: scripts/_render_bf16_vae_recon_large.py and src/data/anytop_rot6d_fk.py, plus only directly necessary related files.

Claims / review points:
1. In scripts/_render_bf16_vae_recon_large.py, the rot6d-FK path removed a double-root-rotation line equivalent to `rqj[:, 0] = qm(qn(rq), rqj[:, 0])`. Is the patched script rot() equivalent to src/data/anytop_rot6d_fk.py recover_from_bvh_rot_np? Is there another double-rotation in the script?
2. Are the verbatim copied helpers r6m/qft/qm/qn/qmat/pg/rqp/ric/rot likely to drift from src behavior? Should the script import src recovery instead of copying? Note ric is the position-channel RIC path.
3. Is the PIL rendering logic coherent: oblique projection u=x-0.36z, v=y-0.22z; per-frame per-panel root-centering; unified transform computed from root-centered sampled frames? Does root-centering hide root trajectory error?
4. Is VAE recon inference correct for a bf16-autocast-trained checkpoint: load_anytop13_vae, vae(batch, sample=False), de-normalize pred = out*(std+floor)+mean, running on CPU/fp32? Are weights fp32 in checkpoint?
5. Seal issue: if GT self-check FK(gt) vs RIC(gt) is 0 but all three panels show the same vertical long-line disorder, is that more likely data raw/position-channel issue than FK double or renderer? What commands/metrics would locate it?

Instructions:
- Read only scoped files and directly necessary related files.
- Output concise findings with file line references and commands used.
- Separate verified facts from inferences.
hook: SessionStart
hook: SessionStart Completed
codex
I’ll use the repo-audit skill because this is a code-truth verification request. I’ll keep the read surface to the two named files plus the loader/model definitions needed to validate the inference path.
exec
/usr/bin/bash -lc "sed -n '1,220p' /iridisfs/home/ts1v23/.agents/skills/repo-audit/SKILL.md" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
---
name: repo-audit
description: "Independent repo-truth verification via codex exec. Codex reads the repo directly — Codex does not control what it sees. Use when user says \"audit\", \"verify claims\", \"check the repo\", \"hostile review\", \"repo audit\", or wants independent evidence-first verification."
argument-hint: [claims-or-scope-to-verify]
allowed-tools: Bash(*), Read, Grep, Glob, Write, Edit
---

# Repo Audit: Independent Repo-Truth Verification

Verify claims against repo evidence for: **$ARGUMENTS**

> **Routing**: This skill always uses `codex exec` (review_intent=verify, reviewer_backend=exec).
> See `docs/REVIEWER_ROUTING_POLICY.md` for the full routing policy.

## Overview

This is the reusable hostile-audit primitive for ARIS. It invokes `codex exec` so that the external reviewer (GPT) reads the repository directly. Codex does not curate or filter context — the reviewer decides what to inspect.

This skill is designed to be:
- **Reusable** — callable from any workflow or standalone
- **Evidence-bound** — output separates verified from inferred from unsupported
- **Honest** — blind spots are explicit, not hidden

## Constants

- **AUDIT_MODEL** = uses `codex exec` default model (configured in `~/.codex/config.toml`, typically `gpt-5.4`)
- **AUDIT_TIMEOUT** = 300 seconds (5 minutes)

## Inputs

1. **Claims to verify** — from user argument, `CLAIMS_FROM_RESULTS.md`, `AUTO_REVIEW.md`, or `NARRATIVE_REPORT.md`
2. **Scope** — which files, directories, or artifact types to prioritize (optional — reviewer can explore freely)

If no specific claims are provided, the reviewer audits the repo holistically: reading code, results, logs, and narrative docs, then checking for internal consistency.

## Workflow

### Step 1: Prepare Claims List

If the user provides specific claims, format them as a numbered list. If not, extract claims from available narrative docs.

### Step 2: Check exec Availability

```bash
command -v codex && echo "AVAILABLE" || echo "UNAVAILABLE"
```

If unavailable:
- **Do NOT fall back to MCP silently**
- Report to user:
  ```
  ⚠️ DEGRADED REVIEW: Repo-truth verification was requested via codex exec,
  but exec was unavailable. Used Codex MCP fallback on curated context.
  This is NOT equivalent to an independent repo audit.
  ```
- If user accepts degraded mode, proceed with MCP and label all outputs as DEGRADED

### Step 3: Execute Hostile Audit

```bash
codex exec "$(cat <<'PROMPT'
You are an independent auditor. Your job is to verify claims against this
repository's actual code, data, logs, and artifacts. Trust NOTHING the
author (Codex) tells you — verify everything yourself.

## Claims to verify:
[numbered list of claims]

## Instructions:
1. Read the experiment code, training scripts, and evaluation scripts
2. Read result files (JSON, CSV, logs) and verify reported numbers
3. Check if evaluation metrics are computed correctly
4. Look for cherry-picked results, missing seeds, or suspicious config choices
5. Read narrative docs (NARRATIVE_REPORT.md, AUTO_REVIEW.md) and cross-check
   each factual statement against the actual repo artifacts
6. Check for discrepancies between what the code does and what the docs claim

## Output — use this EXACT structure:

### Verification Report

- **Review intent**: verify
- **Backend requested**: exec
- **Backend used**: exec
- **Status**: FULL

### Evidence Inspected
- Files read: [list every file you opened]
- Logs inspected: [list]
- Artifact types: [JSON, CSV, .tar, wandb, etc.]
- Commands executed: [list shell commands you ran]

### Verified Findings
[Claims you independently confirmed — cite the file and line/value]

### Inferred Findings
[Claims consistent with evidence but not independently recomputed]

### Unsupported or Uncertain
[Claims lacking evidence or contradicting evidence — cite the discrepancy]

### Not Checked / Blind Spots
[What you did NOT inspect — be explicit]

### Overall Assessment
- Trustworthiness: [HIGH / MEDIUM / LOW]
- Critical issues: [list if any]
- Recommendation: [proceed / investigate further / stop and fix]

Be adversarial. Trust nothing. Cite everything.
PROMPT
)" --skip-git-repo-check 2>&1
```

### Step 4: Parse and Save Output

1. Save the full raw output to `REPO_AUDIT.md` in the project root
2. Extract structured fields: trustworthiness, critical issues, verified/unsupported claims
3. If critical issues found, present them prominently to the user

### Step 5: Report

Present results to user with:
- Number of claims verified / inferred / unsupported / not checked
- Any critical discrepancies
- The full `REPO_AUDIT.md` path for reference

## Degraded Mode (MCP Fallback)

If exec is unavailable and user accepts degraded mode:

```
mcp__codex__codex:
  config: {"model_reasoning_effort": "xhigh"}
  prompt: |
    [same prompt as above, but prefixed with:]

    NOTE: This review is running in DEGRADED mode. You are receiving
    curated context from Codex, not reading the repo directly.
    Flag any claims you cannot independently verify.

    [Codex pastes relevant file contents, logs, results]
```

All outputs must include:
```
⚠️ DEGRADED: This audit used Codex MCP on curated context, not codex exec.
It is NOT equivalent to an independent repo audit.
```

## Key Rules

- **Never present MCP fallback as equivalent to exec audit** — always label DEGRADED
- **Save full raw output** — do not summarize or truncate the reviewer's response
- **Cite evidence** — every finding must reference specific files, lines, or values
- **Explicit blind spots** — state what was NOT checked
- **No silent downgrade** — if exec fails, report it loudly

codex
The skill’s default audit path is `codex exec`; I’ll use it as a second checker, but I’m not going to let it widen the scope. I’m also reading the files myself so the final answer is tied to exact line numbers and commands.
exec
/usr/bin/bash -lc "nl -ba scripts/_render_bf16_vae_recon_large.py | sed -n '1,280p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
     1	"""bf16 VAE recon QA — 3-panel LARGE GIF: GT_RIC | PRED_RIC | PRED_FK.
     2	
     3	Learns the AnyTop `render_rot6d_pose_compare.py` rendering style (user-requested
     4	2026-06-03, "渲染太小，学 anytop 的大图方式"):
     5	  - PIL oblique projection: u = x - 0.36*z, v = y - 0.22*z
     6	  - per-frame root-centered (animal stays centered, not shrunk by trajectory)
     7	  - 900x760 / panel, ground grid + axes + root trail, GIF (48 frames, 12 fps)
     8	
     9	recon logic = _render_vae_recon_rot6d_vs_pos.py (verbatim ric/rot recover funcs,
    10	GT self-check <1%bbox verified). VAE forward (fp32 — the bf16-autocast-trained
    11	ckpt stores fp32 weights, so fp32 inference reproduces it) → pred 13ch:
    12	  GT_RIC  = ric(gt)   ch0:3 of the GT clip          (truth, grey)
    13	  PRED_RIC= ric(pred) recon position channel 0:3    (blue)
    14	  PRED_FK = rot(pred) recon rot6d channel 3:9 + FK   (orange)
    15	
    16	Run on compute node (offline OK, local data); default CPU so it never grabs a
    17	training GPU:
    18	  python scripts/_render_bf16_vae_recon_large.py --ckpt <bf16_vae>/best_model.pt
    19	"""
    20	import sys
    21	import importlib.util
    22	import argparse
    23	from pathlib import Path
    24	import numpy as np
    25	import torch
    26	
    27	ROOT = Path(__file__).resolve().parents[1]
    28	sys.path.insert(0, str(ROOT))
    29	
    30	from PIL import Image, ImageDraw, ImageFont  # noqa: E402
    31	
    32	
    33	# ===== verbatim recover funcs (from _render_vae_recon_rot6d_vs_pos.py, <1%bbox) =====
    34	def r6m(c):
    35	    x = c[..., 0:3] / np.linalg.norm(c[..., 0:3], axis=-1, keepdims=True)
    36	    z = np.cross(x, c[..., 3:6], axis=-1); z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    37	    y = np.cross(z, x, axis=-1)
    38	    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    39	
    40	
    41	def qft(ts):
    42	    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]
    43	    q0 = np.sqrt(((d0 + d1 + d2 + 1) / 4).clip(0, None)); q1 = np.sqrt(((d0 - d1 - d2 + 1) / 4).clip(0, None))
    44	    q2 = np.sqrt(((-d0 + d1 - d2 + 1) / 4).clip(0, None)); q3 = np.sqrt(((-d0 - d1 + d2 + 1) / 4).clip(0, None))
    45	    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3); c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    46	    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3); c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
    47	    q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2]); q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0]); q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])
    48	    q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2]); q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1]); q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])
    49	    q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0]); q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1]); q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])
    50	    q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1]); q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2]); q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])
    51	    return np.stack([q0, q1, q2, q3], axis=-1)
    52	
    53	
    54	def qm(s, o):
    55	    s, o = np.broadcast_arrays(s, o)
    56	    q0, q1, q2, q3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    57	    r0, r1, r2, r3 = o[..., 0], o[..., 1], o[..., 2], o[..., 3]
    58	    return np.stack([r0*q0-r1*q1-r2*q2-r3*q3, r0*q1+r1*q0-r2*q3+r3*q2,
    59	                     r0*q2+r1*q3+r2*q0-r3*q1, r0*q3-r1*q2+r2*q1+r3*q0], axis=-1)
    60	
    61	
    62	def qn(q):
    63	    return q * np.array([1.0, -1.0, -1.0, -1.0])
    64	
    65	
    66	def qmv(q, v):
    67	    vs = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    68	    return qm(q, qm(vs, qn(q)))[..., 1:]
    69	
    70	
    71	def qmat(q):
    72	    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    73	    x2, y2, z2 = qx+qx, qy+qy, qz+qz
    74	    xx, yy, wx, xy, yz = qx*x2, qy*y2, qw*x2, qx*y2, qy*z2
    75	    wy, xz, zz, wz = qw*y2, qx*z2, qz*z2, qw*z2
    76	    m = np.empty(q.shape[:-1] + (3, 3))
    77	    m[..., 0, 0] = 1-(yy+zz); m[..., 0, 1] = xy-wz; m[..., 0, 2] = xz+wy
    78	    m[..., 1, 0] = xy+wz; m[..., 1, 1] = 1-(xx+zz); m[..., 1, 2] = yz-wx
    79	    m[..., 2, 0] = xz-wy; m[..., 2, 1] = yz+wx; m[..., 2, 2] = 1-(xx+yy)
    80	    return m
    81	
    82	
    83	def pg(rq, pos, par):
    84	    F, J = rq.shape[:2]; R = qmat(rq); loc = np.zeros((F, J, 4, 4))
    85	    loc[:, :, :3, :3] = R; loc[:, :, :3, 3] = pos; loc[:, :, 3, 3] = 1
    86	    g = np.zeros((F, J, 4, 4)); g[:, 0] = loc[:, 0]
    87	    for i in range(1, J):
    88	        g[:, i] = np.matmul(g[:, int(par[i])], loc[:, i])
    89	    p = g[:, :, :, 3]; return p[:, :, :3] / p[:, :, 3, None]
    90	
    91	
    92	def rqp(data):
    93	    rq = qft(r6m(data[:, 3:9])); rp = np.zeros(data.shape[:-1] + (3,))
    94	    rp[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
    95	    rp = qmv(qn(rq), rp); rp = np.cumsum(rp, axis=-2); rp[..., 1] = data[..., 1]
    96	    return rq, rp
    97	
    98	
    99	def ric(data):
   100	    rq, rp = rqp(data[..., 0, :]); p = data[..., 1:, :3].copy()
   101	    nq = np.repeat(qn(rq)[..., None, :], p.shape[-2], axis=-2)
   102	    p = qmv(nq, p); p[..., 0] += rp[..., 0:1]; p[..., 2] += rp[..., 2:3]
   103	    return np.concatenate([rp[..., None, :], p], axis=-2)
   104	
   105	
   106	def rot(data, par, off):
   107	    rq, rp = rqp(data[:, 0]); rm = qmat(rq); nrm = r6m(data[..., 1:, 3:9])
   108	    allm = np.concatenate([rm[:, None], nrm], axis=1); allq = qft(allm)
   109	    T, J = allq.shape[:2]; rqj = np.zeros((T, J, 4)); rqj[..., 0] = 1
   110	    for j, p in enumerate(par[1:], 1):
   111	        rqj[:, p] = allq[:, j]
   112	    # double-root-rotation FIX (2026-06-03): REMOVED `rqj[:, 0] = qm(qn(rq), rqj[:, 0])`,
   113	    # which DOUBLE-applied the root global rotation (yaw twice). Mirrors the src fix
   114	    # src/data/anytop_rot6d_fk.py:152 (REMOVED 2026-06-01; FK-vs-RIC absL1 0.65→0.0000).
   115	    # The ch3:9 root token + parent reindex already put the correct root facing in
   116	    # rqj[:,0]; the correction was a spurious second apply. (qm/qn still used by ric path.)
   117	    pos = np.repeat(off[None], T, axis=0).astype(float); pos[:, 0] = rp
   118	    return pg(rqj, pos, par)
   119	
   120	
   121	# ===== PIL render (learned from AnyTop render_rot6d_pose_compare.py) =====
   122	GT_COLOR = (90, 90, 90)        # GT_RIC   grey (truth)
   123	POSE_COLOR = (35, 112, 180)    # PRED_RIC blue (recon pos 0:3)
   124	ROTFK_COLOR = (210, 83, 45)    # PRED_FK  orange (recon rot6d-FK 3:9)
   125	ROOT_COLOR = (18, 18, 18)
   126	TRAIL_COLOR = (150, 150, 150)
   127	GROUND_COLOR = (224, 224, 224)
   128	AXIS_X = (210, 34, 34); AXIS_Y = (30, 150, 55); AXIS_Z = (30, 80, 210)
   129	
   130	
   131	def view_uv(points):
   132	    return points[..., 0] - 0.36 * points[..., 2], points[..., 1] - 0.22 * points[..., 2]
   133	
   134	
   135	def compute_transform(point_sets, size, pad, zoom):
   136	    points = np.concatenate([p.reshape(-1, 3) for p in point_sets if p.size], axis=0)
   137	    u, v = view_uv(points)
   138	    u_min, u_max = float(u.min()), float(u.max()); v_min, v_max = float(v.min()), float(v.max())
   139	    width, height = size
   140	    u_span = max(u_max - u_min, 1e-6); v_span = max(v_max - v_min, 1e-6)
   141	    base = min(width * (1 - 2 * pad) / u_span, height * (1 - 2 * pad) / v_span)
   142	    return base * zoom, (u_min + u_max) * 0.5, (v_min + v_max) * 0.5
   143	
   144	
   145	def project(points, transform, size):
   146	    scale, u_mid, v_mid = transform; width, height = size
   147	    u, v = view_uv(points)
   148	    px = width * 0.5 + (u - u_mid) * scale
   149	    py = height * 0.54 - (v - v_mid) * scale
   150	    return np.stack([px, py], axis=-1)
   151	
   152	
   153	def get_font(size):
   154	    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
   155	              "/usr/share/fonts/dejavu/DejaVuSans.ttf",
   156	              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
   157	              "/usr/share/fonts/liberation/LiberationSans-Regular.ttf"]:
   158	        if Path(c).exists():
   159	            return ImageFont.truetype(c, size)
   160	    return ImageFont.load_default()
   161	
   162	
   163	def draw_axes(draw, size):
   164	    width, height = size; ox = int(width * 0.10); oy = int(height * 0.82)
   165	    length = int(min(width, height) * 0.14)
   166	    draw.line([(ox, oy), (ox + length, oy)], fill=AXIS_X, width=4)
   167	    draw.line([(ox, oy), (ox, oy - length)], fill=AXIS_Y, width=4)
   168	    draw.line([(ox, oy), (ox - int(0.45 * length), oy + int(0.65 * length))], fill=AXIS_Z, width=4)
   169	    lf = get_font(18)
   170	    draw.text((ox + length + 6, oy - 10), "+X", fill=AXIS_X, font=lf)
   171	    draw.text((ox + 5, oy - length - 22), "+Y", fill=AXIS_Y, font=lf)
   172	    draw.text((ox - int(0.45 * length) - 30, oy + int(0.65 * length) - 7), "+Z", fill=AXIS_Z, font=lf)
   173	
   174	
   175	def draw_ground(draw, transform, size, radius):
   176	    for value in np.linspace(-radius, radius, 7):
   177	        for line in [np.array([[-radius, 0.0, value], [radius, 0.0, value]], dtype=float),
   178	                     np.array([[value, 0.0, -radius], [value, 0.0, radius]], dtype=float)]:
   179	            pts = project(line, transform, size)
   180	            draw.line([tuple(pts[0]), tuple(pts[1])], fill=GROUND_COLOR, width=1)
   181	
   182	
   183	def draw_skeleton(draw, positions, parents, transform, size, color, lw, jr, trail):
   184	    if trail is not None and len(trail) > 1:
   185	        t2d = project(trail, transform, size)
   186	        draw.line([tuple(p) for p in t2d], fill=TRAIL_COLOR, width=max(2, lw - 1))
   187	    pts = project(positions, transform, size)
   188	    for j in range(1, len(parents)):
   189	        p = int(parents[j])
   190	        if p >= 0:
   191	            draw.line([tuple(pts[p]), tuple(pts[j])], fill=color, width=lw)
   192	    root = tuple(pts[0]); r = jr
   193	    draw.ellipse((root[0]-r, root[1]-r, root[0]+r, root[1]+r), fill=ROOT_COLOR)
   194	
   195	
   196	def render_panel(positions, parents, fi, transform, size, title, color, lw, jr, axes):
   197	    image = Image.new("RGB", size, "white"); draw = ImageDraw.Draw(image)
   198	    radius = max(2.0, float(np.linalg.norm(positions.reshape(-1, 3)[:, [0, 2]], axis=-1).max()) * 1.05)
   199	    draw_ground(draw, transform, size, radius)
   200	    frame = positions[fi]
   201	    trail = positions[:, 0].copy(); trail[:, 1] = 0.0
   202	    draw_skeleton(draw, frame, parents, transform, size, color, lw, jr, trail)
   203	    if axes:
   204	        draw_axes(draw, size)
   205	    draw.text((18, 16), title, fill=color, font=get_font(24))
   206	    return image
   207	
   208	
   209	def make_frame_3panel(gt, pred_ric, pred_fk, parents, fi, transform, cell, lw, jr):
   210	    W, H = cell
   211	    image = Image.new("RGB", (W * 3, H), "white")
   212	    series_list = [(gt, "GT_RIC (truth)", GT_COLOR, True),
   213	                   (pred_ric, "PRED_RIC recon 0:3", POSE_COLOR, False),
   214	                   (pred_fk, "PRED_FK recon 3:9", ROTFK_COLOR, False)]
   215	    for k, (positions, title, color, axes) in enumerate(series_list):
   216	        s = positions.copy()
   217	        root = s[fi, 0].copy()          # per-panel root-centered (animal stays centered)
   218	        s[..., 0] -= root[0]; s[..., 2] -= root[2]
   219	        panel = render_panel(s, parents, fi, transform, cell, title, color, lw, jr, axes)
   220	        image.paste(panel, (W * k, 0))
   221	    return image
   222	
   223	
   224	# ===== VAE recon =====
   225	from src.data.anytop_dataset import AnyTopDataset, _STD_FLOOR, collate_fn  # noqa: E402
   226	from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402,F401
   227	_spec = importlib.util.spec_from_file_location("aa13", str(ROOT / "scripts" / "animate_anytop13.py"))
   228	aa = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(aa)
   229	
   230	
   231	def sample_indices(length, max_frames):
   232	    if length <= max_frames:
   233	        return list(range(length))
   234	    return sorted(set(int(round(v)) for v in np.linspace(0, length - 1, max_frames)))
   235	
   236	
   237	def main():
   238	    ap = argparse.ArgumentParser()
   239	    ap.add_argument("--ckpt", default=str(ROOT / "runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_model.pt"))
   240	    ap.add_argument("--out", default=str(ROOT / "runs/_bf16_vae_recon_large"))
   241	    ap.add_argument("--data-root", default=str(ROOT / "data/anytop_planet_zoo_clean_L2"))
   242	    ap.add_argument("--device", default="cpu")            # cpu = never grab a training GPU
   243	    ap.add_argument("--cell-width", type=int, default=900)
   244	    ap.add_argument("--cell-height", type=int, default=760)
   245	    ap.add_argument("--max-frames", type=int, default=48)
   246	    ap.add_argument("--fps", type=float, default=12.0)
   247	    ap.add_argument("--zoom", type=float, default=1.15)
   248	    ap.add_argument("--pad", type=float, default=0.06)
   249	    ap.add_argument("--line-width", type=int, default=3)
   250	    ap.add_argument("--joint-radius", type=int, default=4)
   251	    ap.add_argument("--species", action="append", default=[])
   252	    args = ap.parse_args()
   253	
   254	    dev = torch.device(args.device)
   255	    vae, ta = aa.load_anytop13_vae(args.ckpt, dev)
   256	    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
   257	    ds = AnyTopDataset(split="val", val_frac=0.05, seed=42, data_root=args.data_root,
   258	                       num_frames=64, max_joints=144, caption_emb_cache=None)
   259	    want = args.species or ["PZ_Asian_Water_Monitor_Male", "PZ_Komodo_Dragon_Male",
   260	                            "PZ_Saltwater_Crocodile_Male", "PZ_Grey_Seal_Male"]
   261	    cell = (args.cell_width, args.cell_height)
   262	    done = set()
   263	    with torch.no_grad():
   264	        for i in range(len(ds)):
   265	            it = ds[i]; sp = it["object_type"]
   266	            if sp not in want or sp in done:
   267	                continue
   268	            done.add(sp)
   269	            J = int(it["num_joints"]); T = int(it["num_frames"])
   270	            parents = np.asarray([int(p) for p in it["parent_indices"][:J]], dtype=int)
   271	            offsets = np.asarray(it["rest_offsets"], np.float32)[:J].astype(np.float64)
   272	            raw = collate_fn([it]); raw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in raw.items()}
   273	            batch = GraphMotionBatch.from_collate_dict(raw)
   274	            out = vae(batch, sample=False)
   275	            Tv = int(out["frame_mask_recovered"][0].sum().item()); T = min(T, Tv)
   276	            std = raw["anytop_std"][0, :J].cpu().numpy(); mean = raw["anytop_mean"][0, :J].cpu().numpy()
   277	            pred = (out["pred_motion"][0, :T, :J, :].cpu().numpy() * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
   278	            gt = (np.transpose(np.asarray(it["anytop_x"], np.float32), (2, 0, 1))[:T, :J, :]
   279	                  * (std[None] + _STD_FLOOR) + mean[None]).astype(np.float64)
   280	

exec
/usr/bin/bash -lc "nl -ba src/data/anytop_rot6d_fk.py | sed -n '1,260p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
     1	"""AnyTop rot6d FK recovery: recover world-space joint positions from the 13ch
     2	RIFKE encoding via the ROTATION channels (3:9) + bone offsets + parent chain,
     3	i.e. the official `recover_from_bvh_rot_np` path — NOT the position channel
     4	(0:3) RIC path used by `_recover_world_positions`.
     5	
     6	This is a self-contained numpy port (official-derived, PATCHED 2026-06-01) of
     7	the AnyTop/SALAD `recover_from_bvh_rot_np` (+ `recover_root_quat_and_pos_np`)
     8	from the user's planetzoo-anytop-pipeline / truebones motion_process.py.
     9	
    10	PATCH (2026-06-01): the official source applies a root correction
    11	`rot_q[:,0] = -r_rot_quat * rot_q[:,0]` that DOUBLE-applies the root global
    12	rotation (turn yaw twice). REMOVED here — the ch3:9 root/child tokens already
    13	carry the global facing, and the parent reindex already puts the correct root
    14	orientation in rot_q[:,0]. Verified: WITH the correction FK-vs-RIC absL1=0.65
    15	(global-orient sweep ~1.98); WITHOUT it absL1=0.0000 (FK == RIC exactly) on
    16	clean_L2 Saiga AND the 1070 old-truebones largest-rotation clips. The earlier
    17	"<1% bbox" check that called this FK correct used near-idle clips (root barely
    18	rotates) and missed the bug.
    19	
    20	Why a self-contained port: the original lib needs numpy.core.umath_tests
    21	(removed in new numpy) + a heavy BVH/Quaternions/Animation dependency chain that
    22	won't import here. Every operator below re-implements the authoritative source
    23	(Quaternions.__mul__/from_transforms/__neg__/transforms, Animation.positions_
    24	global 4x4 matmul chain, rotation_6d_to_matrix_np) EXCEPT the removed root
    25	correction noted in the PATCH above.
    26	
    27	Inputs use the dataset's own per-item `rest_offsets` + `parent_indices`, which
    28	share the same `new_to_old_perm` joint ordering as `anytop_x` (aligned).
    29	"""
    30	import numpy as np
    31	
    32	
    33	def _rotation_6d_to_matrix_np(c):
    34	    """utils/rotation_conversions.py:536 verbatim. [...,6]->[...,3,3] cols [x,y,z]."""
    35	    x = c[..., 0:3] / np.linalg.norm(c[..., 0:3], axis=-1, keepdims=True)
    36	    z = np.cross(x, c[..., 3:6], axis=-1)
    37	    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    38	    y = np.cross(z, x, axis=-1)
    39	    return np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    40	
    41	
    42	def _quat_from_transforms(ts):
    43	    """Quaternions.from_transforms verbatim (rotation matrix -> wxyz quat)."""
    44	    d0, d1, d2 = ts[..., 0, 0], ts[..., 1, 1], ts[..., 2, 2]
    45	    q0 = np.sqrt(((d0 + d1 + d2 + 1) / 4).clip(0, None))
    46	    q1 = np.sqrt(((d0 - d1 - d2 + 1) / 4).clip(0, None))
    47	    q2 = np.sqrt(((-d0 + d1 - d2 + 1) / 4).clip(0, None))
    48	    q3 = np.sqrt(((-d0 - d1 + d2 + 1) / 4).clip(0, None))
    49	    c0 = (q0 >= q1) & (q0 >= q2) & (q0 >= q3)
    50	    c1 = (q1 >= q0) & (q1 >= q2) & (q1 >= q3)
    51	    c2 = (q2 >= q0) & (q2 >= q1) & (q2 >= q3)
    52	    c3 = (q3 >= q0) & (q3 >= q1) & (q3 >= q2)
    53	    q1[c0] *= np.sign(ts[c0, 2, 1] - ts[c0, 1, 2]); q2[c0] *= np.sign(ts[c0, 0, 2] - ts[c0, 2, 0]); q3[c0] *= np.sign(ts[c0, 1, 0] - ts[c0, 0, 1])
    54	    q0[c1] *= np.sign(ts[c1, 2, 1] - ts[c1, 1, 2]); q2[c1] *= np.sign(ts[c1, 1, 0] + ts[c1, 0, 1]); q3[c1] *= np.sign(ts[c1, 0, 2] + ts[c1, 2, 0])
    55	    q0[c2] *= np.sign(ts[c2, 0, 2] - ts[c2, 2, 0]); q1[c2] *= np.sign(ts[c2, 1, 0] + ts[c2, 0, 1]); q3[c2] *= np.sign(ts[c2, 2, 1] + ts[c2, 1, 2])
    56	    q0[c3] *= np.sign(ts[c3, 1, 0] - ts[c3, 0, 1]); q1[c3] *= np.sign(ts[c3, 2, 0] + ts[c3, 0, 2]); q2[c3] *= np.sign(ts[c3, 2, 1] + ts[c3, 1, 2])
    57	    return np.stack([q0, q1, q2, q3], axis=-1)
    58	
    59	
    60	def _quat_mul(s, o):
    61	    """Quaternions.__mul__ (q*q) verbatim."""
    62	    s, o = np.broadcast_arrays(s, o)
    63	    q0, q1, q2, q3 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    64	    r0, r1, r2, r3 = o[..., 0], o[..., 1], o[..., 2], o[..., 3]
    65	    return np.stack([
    66	        r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3,
    67	        r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2,
    68	        r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1,
    69	        r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0], axis=-1)
    70	
    71	
    72	def _quat_neg(q):
    73	    """Quaternions.__neg__ : conjugate."""
    74	    return q * np.array([1.0, -1.0, -1.0, -1.0])
    75	
    76	
    77	def _quat_mul_vec(q, v):
    78	    """Quaternions.__mul__ (q*vec) verbatim: (q*(vs*-q)).imaginaries, vs=[0,v]."""
    79	    vs = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    80	    return _quat_mul(q, _quat_mul(vs, _quat_neg(q)))[..., 1:]
    81	
    82	
    83	def _quat_to_matrix(q):
    84	    """Quaternions.transforms() verbatim (wxyz -> 3x3)."""
    85	    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    86	    x2, y2, z2 = qx + qx, qy + qy, qz + qz
    87	    xx, yy, wx, xy, yz = qx * x2, qy * y2, qw * x2, qx * y2, qy * z2
    88	    wy, xz, zz, wz = qw * y2, qx * z2, qz * z2, qw * z2
    89	    m = np.empty(q.shape[:-1] + (3, 3))
    90	    m[..., 0, 0] = 1 - (yy + zz); m[..., 0, 1] = xy - wz; m[..., 0, 2] = xz + wy
    91	    m[..., 1, 0] = xy + wz; m[..., 1, 1] = 1 - (xx + zz); m[..., 1, 2] = yz - wx
    92	    m[..., 2, 0] = xz - wy; m[..., 2, 1] = yz + wx; m[..., 2, 2] = 1 - (xx + yy)
    93	    return m
    94	
    95	
    96	def _positions_global(rot_q, positions, parents):
    97	    """Animation.positions_global verbatim: 4x4 local->global matmul chain.
    98	    rot_q [F,J,4]; positions [F,J,3]; parents [J]. Returns global xyz [F,J,3]."""
    99	    F, J = rot_q.shape[:2]
   100	    R = _quat_to_matrix(rot_q)
   101	    loc = np.zeros((F, J, 4, 4))
   102	    loc[:, :, :3, :3] = R
   103	    loc[:, :, :3, 3] = positions
   104	    loc[:, :, 3, 3] = 1.0
   105	    glob = np.zeros((F, J, 4, 4))
   106	    glob[:, 0] = loc[:, 0]
   107	    for i in range(1, J):
   108	        glob[:, i] = np.matmul(glob[:, int(parents[i])], loc[:, i])
   109	    p = glob[:, :, :, 3]
   110	    return p[:, :, :3] / p[:, :, 3, None]
   111	
   112	
   113	def _recover_root_quat_and_pos_np(data):
   114	    """motion_process.py:700 verbatim. data [T,13] (root joint). Returns
   115	    (r_rot_quat [T,4], r_pos [T,3]). Shared by both ric and rot paths."""
   116	    r_rot_quat = _quat_from_transforms(_rotation_6d_to_matrix_np(data[:, 3:9]))
   117	    r_pos = np.zeros(data.shape[:-1] + (3,))
   118	    r_pos[..., 1:, [0, 2]] = data[..., :-1, [9, 11]]
   119	    r_pos = _quat_mul_vec(_quat_neg(r_rot_quat), r_pos)
   120	    r_pos = np.cumsum(r_pos, axis=-2)
   121	    r_pos[..., 1] = data[..., 1]
   122	    return r_rot_quat, r_pos
   123	
   124	
   125	def recover_from_bvh_rot_np(data, parents, offsets):
   126	    """Official `recover_from_bvh_rot_np` (motion_process.py:750) verbatim port.
   127	
   128	    Recover world joint positions from the ROTATION channels (3:9) via FK on the
   129	    parent chain with bone `offsets`, plus the integrated root translation.
   130	
   131	    Args:
   132	      data:    [T, J, 13] RAW (un-normalized) AnyTop motion (FK-ordered joints).
   133	      parents: [J] int, FK order (parents[0] = -1, parents[j] < j).
   134	      offsets: [J, 3] rest-pose bone offsets, SAME joint ordering as `data`.
   135	    Returns:
   136	      [T, J, 3] world-space joint positions.
   137	    """
   138	    data = np.asarray(data, dtype=np.float64)
   139	    parents = np.asarray(parents, dtype=int)
   140	    offsets = np.asarray(offsets, dtype=np.float64)
   141	    T, J, _ = data.shape
   142	    r_rot_quat, r_pos = _recover_root_quat_and_pos_np(data[:, 0])      # [T,4],[T,3]
   143	    r_rot_mat = _quat_to_matrix(r_rot_quat)                            # [T,3,3]
   144	    nonroot_mat = _rotation_6d_to_matrix_np(data[:, 1:, 3:9])          # [T,J-1,3,3]
   145	    all_mat = np.concatenate([r_rot_mat[:, None], nonroot_mat], axis=1)  # [T,J,3,3]
   146	    all_q_hml = _quat_from_transforms(all_mat)                         # [T,J,4]
   147	    # parent reindex (motion_process.py:758-759): rotations[:,p] = hml[:,j]
   148	    rot_q = np.zeros((T, J, 4)); rot_q[..., 0] = 1.0
   149	    for j, p in enumerate(parents[1:], 1):
   150	        rot_q[:, p] = all_q_hml[:, j]
   151	    # NOTE: the official recover_from_bvh_rot_np applies a root correction here
   152	    # (rot_q[:,0] = -r_rot_quat * rot_q[:,0]). REMOVED 2026-06-01 — it
   153	    # DOUBLE-applies the root global rotation. Proof (RIC path = ground truth):
   154	    # WITH it,   FK-vs-RIC absL1=0.6522 (global-orient sweep ratio ~1.98);
   155	    # WITHOUT it, FK-vs-RIC absL1=0.0000 (FK == RIC exactly).
   156	    # Verified on clean_L2 Saiga AND the 1070 old-truebones largest-rotation
   157	    # clips (Parrot/Bird CircleFly 720/714 deg, Trex turn_180 396 deg, ...): all
   158	    # absL1=0.0000 once removed. rot_q[:,0] keeps its reindexed value (the root
   159	    # child's rotation from the parent reindex), which already carries the
   160	    # correct root orientation — the correction was a spurious second apply.
   161	    pos = np.repeat(offsets[None], T, axis=0).astype(np.float64)       # [T,J,3]
   162	    pos[:, 0] = r_pos
   163	    return _positions_global(rot_q, pos, parents).astype(np.float32)

exec
/usr/bin/bash -lc 'command -v codex && codex --version' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
/home/ts1v23/.nvm/versions/node/v22.12.0/bin/codex
codex-cli 0.135.0

exec
/usr/bin/bash -lc "pwd && git status --short && rg --files scripts src | rg '(_render_bf16_vae_recon_large\\.py|anytop_rot6d_fk\\.py|vae|AnyTop|normal|norm|dataset)'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 110ms:
scripts/animate_graph_vae.py
scripts/eval_graph_vae.py
scripts/_launch_bf16_vae_8card_xnode.sh
scripts/train_graph_vae.py
scripts/_codex_bf16_vae_brief.md
src/models/slot_norm.py
scripts/_deploy_train_graph_vae.sh
scripts/_render_vae_qa_cont1.sh
scripts/_oneshot_vae_recon_specific.py
scripts/_render_bf16_vae_recon_large.py
scripts/_diag_vae_nan.py
src/models/graph_salad/vae.py
scripts/_render_vae_recon_rot6d_vs_pos.py
scripts/self_test_graph_vae.py

codex
The first pass shows the script carries local copies of the FK and RIC helpers, while `src/data/anytop_rot6d_fk.py` has the canonical patched FK. I’m checking the remaining render/inference block and the directly imported loader/model paths now.
exec
/usr/bin/bash -lc "nl -ba scripts/_render_bf16_vae_recon_large.py | sed -n '280,420p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
   280	
   281	            gt_ric = ric(gt)                          # GT_RIC   (truth, grey)
   282	            pred_ric = ric(pred)                      # PRED_RIC (recon pos 0:3, blue)
   283	            pred_fk = rot(pred, parents, offsets)     # PRED_FK  (recon rot6d-FK 3:9, orange)
   284	            gt_rot = rot(gt, parents, offsets)
   285	            gtbbox = np.linalg.norm(gt_ric.reshape(-1, 3).max(0) - gt_ric.reshape(-1, 3).min(0))
   286	            gtchk = np.linalg.norm(gt_ric - gt_rot, axis=-1).mean() / gtbbox * 100
   287	            recon_err = np.linalg.norm(gt_ric - pred_ric, axis=-1).mean() / gtbbox * 100
   288	            print(f"{sp} J={J} T={T} | GT_selfchk={gtchk:.2f}% (<2 ok, renderer faithful) | "
   289	                  f"recon_err(GT-vs-PRED_RIC)={recon_err:.2f}%bbox", flush=True)
   290	
   291	            # ground-normalize each (y min -> 0) so all panels share the floor plane
   292	            for arr in (gt_ric, pred_ric, pred_fk):
   293	                arr[..., 1] -= arr[..., 1].min()
   294	
   295	            idxs = sample_indices(T, args.max_frames)
   296	            # unified transform across all 3 panels — feed each panel's root-centered
   297	            # sampled frames (matches make_frame_3panel's per-panel root-centering)
   298	            ps = []
   299	            for arr in (gt_ric, pred_ric, pred_fk):
   300	                c = arr.copy(); roots = c[:, 0].copy()
   301	                c[..., 0] -= roots[:, None, 0]; c[..., 2] -= roots[:, None, 2]
   302	                ps += [c[k] for k in idxs]
   303	            transform = compute_transform(ps, cell, args.pad, args.zoom)
   304	
   305	            frames = [make_frame_3panel(gt_ric, pred_ric, pred_fk, parents, k, transform, cell,
   306	                                        args.line_width, args.joint_radius) for k in idxs]
   307	            out_path = out_dir / f"{sp}_recon_3panel_large.gif"
   308	            dur = int(round(1000.0 / max(args.fps, 1e-6)))
   309	            frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
   310	            print(f"  saved {out_path} ({len(frames)}f, {cell[0]*3}x{cell[1]})", flush=True)
   311	    print("DONE", flush=True)
   312	
   313	
   314	if __name__ == "__main__":
   315	    main()

exec
/usr/bin/bash -lc 'rg -n "def load_anytop13_vae|_STD_FLOOR|anytop_std|anytop_mean|pred_motion|frame_mask_recovered|GraphMotionBatch|class GraphMotionVAE|def forward" scripts/animate_anytop13.py src/data/anytop_dataset.py src/models/graph_salad -S' in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
src/data/anytop_dataset.py:32:    [T, 4] foot_contact key is kept zero-filled for GraphMotionBatch schema
src/data/anytop_dataset.py:41:      anytop_mean [J, 13], anytop_std [J, 13] (raw, un-normalized)
src/data/anytop_dataset.py:78:_STD_FLOOR = 1e-6  # matches AnyTop's `std += 1e-6` stability constant
src/data/anytop_dataset.py:454:    """AnyTop truebones_processed -> GraphMotionBatch-compatible samples.
src/data/anytop_dataset.py:819:        std_safe = std + _STD_FLOOR
src/data/anytop_dataset.py:993:            # ---- GraphMotionBatch-compatible padded tensors ----
src/data/anytop_dataset.py:1025:            # ---- AnyTop-specific extras (NOT validated by GraphMotionBatch) ----
src/data/anytop_dataset.py:1030:            "anytop_mean": torch.from_numpy(mean_padded),                  # [Jm, 13]
src/data/anytop_dataset.py:1031:            "anytop_std": torch.from_numpy(std_padded),                    # [Jm, 13]
src/data/anytop_dataset.py:1046:    resulting dict feeds GraphMotionBatch.from_collate_dict() unchanged.
scripts/animate_anytop13.py:3:The anytop13 decoder outputs `pred_motion [B,T,J,13]` in AnyTop's NORMALIZED
scripts/animate_anytop13.py:6:  pred_motion (normalized 13ch)
scripts/animate_anytop13.py:40:    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
scripts/animate_anytop13.py:43:from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402
scripts/animate_anytop13.py:46:def load_anytop13_vae(ckpt_path: str, dev: torch.device):
scripts/animate_anytop13.py:160:            batch = GraphMotionBatch.from_collate_dict(raw)
scripts/animate_anytop13.py:172:            T_valid = int(out["frame_mask_recovered"][0].sum().item())
scripts/animate_anytop13.py:176:            # anytop_mean/std are de-norm stats — only needed for visualization,
scripts/animate_anytop13.py:178:            # typed GraphMotionBatch).
scripts/animate_anytop13.py:179:            std = raw["anytop_std"][0, :J].cpu().numpy()    # [J, 13]
scripts/animate_anytop13.py:180:            mean = raw["anytop_mean"][0, :J].cpu().numpy()  # [J, 13]
scripts/animate_anytop13.py:181:            pred_norm = out["pred_motion"][0, :T, :J, :].cpu().numpy()  # [T, J, 13]
scripts/animate_anytop13.py:182:            pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
scripts/animate_anytop13.py:191:                gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]
scripts/animate_anytop13.py:200:                gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]
src/models/graph_salad/batch.py:1:"""GraphMotionBatch — typed wrapper around UnifiedMotionDataset.collate_fn output.
src/models/graph_salad/batch.py:9:Plan §3 originally proposed a GraphMotionBatch dataclass with custom collate;
src/models/graph_salad/batch.py:79:class GraphMotionBatch:
src/models/graph_salad/batch.py:82:    Constructed via ``GraphMotionBatch.from_collate_dict(d)`` — does NOT copy
src/models/graph_salad/batch.py:135:    anytop_mean: Optional[torch.Tensor] = None             # [B, J_max, 13] raw de-norm mean
src/models/graph_salad/batch.py:136:    anytop_std: Optional[torch.Tensor] = None              # [B, J_max, 13] raw de-norm std
src/models/graph_salad/batch.py:141:    def from_collate_dict(cls, d: dict[str, Any]) -> "GraphMotionBatch":
src/models/graph_salad/batch.py:164:                f"GraphMotionBatch: collate dict missing required keys: {missing}"
src/models/graph_salad/batch.py:171:                f"GraphMotionBatch: 'motion_features' must be torch.Tensor, "
src/models/graph_salad/batch.py:176:                f"GraphMotionBatch: 'motion_features' must have rank >= 1 to "
src/models/graph_salad/batch.py:182:                f"GraphMotionBatch: batch size must be > 0, got B={B}"
src/models/graph_salad/batch.py:192:                    f"GraphMotionBatch: '{key}' must be torch.Tensor, "
src/models/graph_salad/batch.py:197:                    f"GraphMotionBatch: '{key}' must have rank {expected_rank}, "
src/models/graph_salad/batch.py:202:                    f"GraphMotionBatch: '{key}' batch dim {t.shape[0]} "
src/models/graph_salad/batch.py:207:                    f"GraphMotionBatch: '{key}' last-dim must be {last_dim}, "
src/models/graph_salad/batch.py:212:                    f"GraphMotionBatch: '{key}' dtype must be {expected_dtype}, "
src/models/graph_salad/batch.py:217:                    f"GraphMotionBatch: '{key}' device {t.device} != "
src/models/graph_salad/batch.py:223:                    f"GraphMotionBatch: '{key}' contains NaN or Inf "
src/models/graph_salad/batch.py:232:                    f"GraphMotionBatch: '{key}' must be square (J_max x J_max), "
src/models/graph_salad/batch.py:241:                    f"GraphMotionBatch: batched-scalar '{key}' must be torch.Tensor "
src/models/graph_salad/batch.py:247:                    f"GraphMotionBatch: batched-scalar '{key}' shape must be ({B},), "
src/models/graph_salad/batch.py:252:                    f"GraphMotionBatch: batched-scalar '{key}' dtype must be "
src/models/graph_salad/batch.py:257:                    f"GraphMotionBatch: batched-scalar '{key}' device {t.device} "
src/models/graph_salad/batch.py:268:                    f"GraphMotionBatch: '{key}' T-dim {d[key].shape[1]} "
src/models/graph_salad/batch.py:275:                    f"GraphMotionBatch: '{key}' J-dim {d[key].shape[1]} "
src/models/graph_salad/batch.py:281:                    f"GraphMotionBatch: '{key}' J-dim {d[key].shape[2]} "
src/models/graph_salad/batch.py:292:                f"GraphMotionBatch: num_joints out of range [1, J_max={J_max_val}], "
src/models/graph_salad/batch.py:298:                f"GraphMotionBatch: num_joints {nj.tolist()} != "
src/models/graph_salad/batch.py:306:                f"GraphMotionBatch: num_frames out of range [1, T_max={T_max_val}], "
src/models/graph_salad/batch.py:312:                f"GraphMotionBatch: num_frames {nf.tolist()} != "
src/models/graph_salad/batch.py:322:                f"GraphMotionBatch: fps contains NaN or Inf, got {fps_t.tolist()}"
src/models/graph_salad/batch.py:326:                f"GraphMotionBatch: fps must be > 0, got {fps_t.tolist()}"
src/models/graph_salad/batch.py:350:                    f"GraphMotionBatch: '{key}' must be list, got {type(v).__name__}"
src/models/graph_salad/batch.py:354:                    f"GraphMotionBatch: list '{key}' length {len(v)} != batch size {B}"
src/models/graph_salad/batch.py:360:                        f"GraphMotionBatch: '{key}'[{i}] must be list, "
src/models/graph_salad/batch.py:369:                        f"GraphMotionBatch: '{key}'[{i}] length {len(inner)} "
src/models/graph_salad/batch.py:378:                            f"GraphMotionBatch: '{key}'[{i}][{j}] is bool; "
src/models/graph_salad/batch.py:384:                            f"GraphMotionBatch: '{key}'[{i}][{j}] type "
src/models/graph_salad/batch.py:400:                            f"GraphMotionBatch: parent_indices[{i}] is not a "
src/models/graph_salad/batch.py:407:                            f"GraphMotionBatch: parent_indices[{i}] violates "
src/models/graph_salad/batch.py:422:                    f"GraphMotionBatch: joint_mask[{b}] is not contiguous "
src/models/graph_salad/batch.py:427:                    f"GraphMotionBatch: joint_mask[{b}] has True in padded "
src/models/graph_salad/batch.py:433:                    f"GraphMotionBatch: frame_mask[{b}] is not contiguous "
src/models/graph_salad/batch.py:438:                    f"GraphMotionBatch: frame_mask[{b}] has True in padded "
src/models/graph_salad/batch.py:455:                    f"GraphMotionBatch: adjacency[{b},:{nj_b},:{nj_b}] has "
src/models/graph_salad/batch.py:461:                    f"GraphMotionBatch: adjacency[{b},:{nj_b},:{nj_b}] not "
src/models/graph_salad/batch.py:467:                    f"GraphMotionBatch: adjacency[{b}] has non-zero diagonal "
src/models/graph_salad/batch.py:478:                    f"GraphMotionBatch: adjacency[{b},:{nj_b},:{nj_b}] does "
src/models/graph_salad/batch.py:486:                        f"GraphMotionBatch: adjacency[{b}] has non-zero "
src/models/graph_salad/batch.py:494:                    f"GraphMotionBatch: '{key}' must be list, got {type(v).__name__}"
src/models/graph_salad/batch.py:498:                    f"GraphMotionBatch: list '{key}' length {len(v)} != batch size {B}"
src/models/graph_salad/batch.py:503:                        f"GraphMotionBatch: '{key}'[{i}] must be str, "
src/models/graph_salad/batch.py:517:            ("anytop_mean",            3, (J_max_val, 13)),
src/models/graph_salad/batch.py:518:            ("anytop_std",             3, (J_max_val, 13)),
src/models/graph_salad/batch.py:526:                    f"GraphMotionBatch: optional '{key}' must be torch.Tensor, "
src/models/graph_salad/batch.py:531:                    f"GraphMotionBatch: optional '{key}' must have shape "
src/models/graph_salad/batch.py:536:                    f"GraphMotionBatch: optional '{key}' dtype must be float32, "
src/models/graph_salad/batch.py:541:                    f"GraphMotionBatch: optional '{key}' device {t.device} != "
src/models/graph_salad/batch.py:546:                    f"GraphMotionBatch: optional '{key}' contains NaN or Inf"
src/models/graph_salad/batch.py:554:                    f"GraphMotionBatch: optional 'has_text' must be torch.Tensor, "
src/models/graph_salad/batch.py:559:                    f"GraphMotionBatch: optional 'has_text' must be [{B}] bool, "
src/models/graph_salad/batch.py:564:                    f"GraphMotionBatch: optional 'has_text' device {ht.device} "
src/models/graph_salad/batch.py:597:            anytop_mean=d.get("anytop_mean"),
src/models/graph_salad/batch.py:598:            anytop_std=d.get("anytop_std"),
src/models/graph_salad/vae.py:60:class GraphMotionVAE(nn.Module):
src/models/graph_salad/vae.py:347:    def encode(self, batch: "GraphMotionBatch", sample: bool | None = None) -> dict:
src/models/graph_salad/vae.py:518:    def encode_skeleton_only(self, batch: "GraphMotionBatch") -> dict:
src/models/graph_salad/vae.py:600:    def decode(self, encode_out: dict, batch: "GraphMotionBatch") -> dict:
src/models/graph_salad/vae.py:604:        anytop13: → direct 13ch regression → pred_motion [B, T, J, 13].
src/models/graph_salad/vae.py:605:        The return dict always carries pred_pos/pred_vel/pred_motion; the
src/models/graph_salad/vae.py:639:            frame_mask_recovered = frame_mask_lat.repeat_interleave(
src/models/graph_salad/vae.py:648:                frame_mask_recovered,
src/models/graph_salad/vae.py:661:                frame_mask_recovered = unpool_out["frame_mask_up"]
src/models/graph_salad/vae.py:669:                frame_mask_recovered = frame_mask_lat.repeat_interleave(
src/models/graph_salad/vae.py:681:                frame_mask_recovered,
src/models/graph_salad/vae.py:701:                    frame_mask_recovered,
src/models/graph_salad/vae.py:704:        fm_b = frame_mask_recovered[:, :, None, None].to(feats.dtype)
src/models/graph_salad/vae.py:712:            pred_motion = torch.cat([out_root, out_nonroot], dim=2)            # [B,T,J,13]
src/models/graph_salad/vae.py:713:            pred_motion = pred_motion * fm_b * jm_b
src/models/graph_salad/vae.py:715:                "pred_motion": pred_motion,   # [B, T, J, 13]
src/models/graph_salad/vae.py:718:                "frame_mask_recovered": frame_mask_recovered,
src/models/graph_salad/vae.py:726:                   batch.joint_mask, frame_mask_recovered)
src/models/graph_salad/vae.py:746:            "pred_motion": None,
src/models/graph_salad/vae.py:747:            "frame_mask_recovered": frame_mask_recovered,
src/models/graph_salad/vae.py:750:    def forward(self, batch: "GraphMotionBatch", sample: bool | None = None) -> dict:
src/models/graph_salad/losses.py:515:    pred_motion: torch.Tensor,          # [B, T, J, 13] decoder output (normalized)
src/models/graph_salad/losses.py:535:    if pred_motion.shape != gt_motion.shape:
src/models/graph_salad/losses.py:537:            f"compute_total_loss_13ch: pred_motion {tuple(pred_motion.shape)} != "
src/models/graph_salad/losses.py:540:    if pred_motion.dim() != 4 or pred_motion.shape[-1] != 13:
src/models/graph_salad/losses.py:542:            f"compute_total_loss_13ch: pred_motion must be [B,T,J,13], "
src/models/graph_salad/losses.py:543:            f"got {tuple(pred_motion.shape)}"
src/models/graph_salad/losses.py:558:    losses["pos"] = _masked_group_l1(pred_motion, gt_motion, slice(0, 3),
src/models/graph_salad/losses.py:560:    losses["rot"] = _masked_group_l1(pred_motion, gt_motion, slice(3, 9),
src/models/graph_salad/losses.py:562:    losses["vel"] = _masked_group_l1(pred_motion, gt_motion, slice(9, 12),
src/models/graph_salad/losses.py:565:        pred_motion[..., 12], foot_contact_per_joint, joint_mask, frame_mask
src/models/graph_salad/losses.py:571:            (), device=pred_motion.device, dtype=pred_motion.dtype
src/models/graph_salad/losses.py:577:    total = torch.zeros((), device=pred_motion.device, dtype=pred_motion.dtype)
src/models/graph_salad/losses.py:611:_ANYTOP_STD_FLOOR = 1e-6  # == anytop_dataset._STD_FLOOR; denorm must invert
src/models/graph_salad/losses.py:612:                          # dataset's normed = (raw-mean)/(std+_STD_FLOOR).
src/models/graph_salad/losses.py:615:def _denorm_13ch(motion_norm, anytop_mean, anytop_std):
src/models/graph_salad/losses.py:618:    motion_norm [B,T,J,13] normalized; anytop_mean/std [B,J,13] raw stats.
src/models/graph_salad/losses.py:622:    mean = anytop_mean.unsqueeze(1)  # [B,1,J,13]
src/models/graph_salad/losses.py:623:    std = anytop_std.unsqueeze(1)    # [B,1,J,13]
src/models/graph_salad/losses.py:624:    return motion_norm * (std + _ANYTOP_STD_FLOOR) + mean
src/models/graph_salad/losses.py:629:    pred_motion: torch.Tensor,   # [B,T,J,13] normalized decoder output
src/models/graph_salad/losses.py:631:    anytop_mean: torch.Tensor,   # [B,J,13] raw de-norm mean
src/models/graph_salad/losses.py:632:    anytop_std: torch.Tensor,    # [B,J,13] raw de-norm std
src/models/graph_salad/losses.py:653:    if anytop_mean is None or anytop_std is None:
src/models/graph_salad/losses.py:655:            "compute_world_geometry_terms requires batch.anytop_mean / "
src/models/graph_salad/losses.py:656:            "anytop_std (loss_mode=anytop13_world_geometry needs de-norm stats)"
src/models/graph_salad/losses.py:658:    pred_raw = _denorm_13ch(pred_motion, anytop_mean, anytop_std)  # [B,T,J,13]
src/models/graph_salad/losses.py:659:    gt_raw = _denorm_13ch(gt_motion, anytop_mean, anytop_std)
src/models/graph_salad/losses.py:691:    pred_motion: torch.Tensor,      # [B,T,J,13] normalized
src/models/graph_salad/losses.py:693:    anytop_mean: torch.Tensor,      # [B,J,13]
src/models/graph_salad/losses.py:694:    anytop_std: torch.Tensor,       # [B,J,13]
src/models/graph_salad/losses.py:698:    frame_mask: torch.Tensor,       # [B,T] bool — use frame_mask_recovered
src/models/graph_salad/losses.py:722:    if anytop_mean is None or anytop_std is None:
src/models/graph_salad/losses.py:724:            "compute_world_rot6d_fk_terms requires batch.anytop_mean / "
src/models/graph_salad/losses.py:725:            "anytop_std (loss_mode=anytop13_world_rot6d_fk needs de-norm stats)")
src/models/graph_salad/losses.py:726:    pred_raw = _denorm_13ch(pred_motion, anytop_mean, anytop_std)  # [B,T,J,13]
src/models/graph_salad/losses.py:727:    gt_raw = _denorm_13ch(gt_motion, anytop_mean, anytop_std)
src/models/graph_salad/denoiser_stub.py:65:    def forward(
src/models/graph_salad/attention.py:54:                                           (GraphMotionBatch, pool_dynamic,
src/models/graph_salad/attention.py:126:    def forward(
src/models/graph_salad/pool_edge_segment.py:76:    # FK ordering). The VAE path enforces this via GraphMotionBatch, but direct
src/models/graph_salad/pool_edge_segment.py:434:    def forward(
src/models/graph_salad/denoiser.py:69:    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
src/models/graph_salad/denoiser.py:101:    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
src/models/graph_salad/denoiser.py:131:    def forward(
src/models/graph_salad/denoiser.py:272:    def forward(
src/models/graph_salad/__init__.py:11:from .batch import GraphMotionBatch
src/models/graph_salad/__init__.py:60:    "GraphMotionBatch",
src/models/graph_salad/pool_dynamic.py:28:    fine graph satisfies it (which GraphMotionBatch already validates).
src/models/graph_salad/pool_dynamic.py:781:    def forward(
src/models/graph_salad/pool_deterministic.py:533:    def forward(
src/models/graph_salad/unpool.py:61:    def forward(

exec
/usr/bin/bash -lc "rg -n \"recover_from_bvh_rot_np|recover_root|_recover_world_positions|rotation_6d|rot_q\\[:,0\\]|rqj\\[:, 0\\]|qm\\(qn\\(rq\\)|_quat_mul\\(_quat_neg\" scripts src -S" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
scripts/_diag_oldset_fk_variants.py:17:from src.data.anytop_dataset import _recover_world_positions  # noqa
scripts/_diag_oldset_fk_variants.py:19:    _recover_root_quat_and_pos_np, _rotation_6d_to_matrix_np, _quat_from_transforms,
scripts/_diag_oldset_fk_variants.py:20:    _quat_to_matrix, _positions_global, recover_from_bvh_rot_np)
scripts/_diag_oldset_fk_variants.py:33:    m = _rotation_6d_to_matrix_np(raw[:, 0, 3:9])  # [T,3,3]
scripts/_diag_oldset_fk_variants.py:66:    ric = _recover_world_positions(raw.astype(np.float32))
scripts/_diag_oldset_fk_variants.py:67:    fkA = recover_from_bvh_rot_np(raw, par, off)
scripts/_diag_oldset_fk_variants.py:69:    r_rot_quat, r_pos = _recover_root_quat_and_pos_np(raw[:, 0])
scripts/_diag_oldset_fk_variants.py:71:    nonroot_mat = _rotation_6d_to_matrix_np(raw[:, 1:, 3:9])
scripts/_render_bf16_vae_recon_large.py:112:    # double-root-rotation FIX (2026-06-03): REMOVED `rqj[:, 0] = qm(qn(rq), rqj[:, 0])`,
src/models/graph_salad/world_recovery.py:3:A torch port of `src.data.anytop_dataset._recover_world_positions` (numpy/scipy)
src/models/graph_salad/world_recovery.py:14:  src.data.anytop_dataset._recover_world_positions(x) to ~1e-4 on real clips.
src/models/graph_salad/world_recovery.py:22:The 6D->matrix convention is identical to numpy `_rotation_6d_to_matrix_np` and
src/models/graph_salad/world_recovery.py:34:    `_rotation_6d_to_matrix_np` and `treeik_decoder.rot6d_to_matrix` bit-for-bit
src/models/graph_salad/world_recovery.py:53:    Mirrors numpy `_recover_world_positions` (anytop_dataset.py:282) exactly;
src/models/graph_salad/losses.py:646:    recover_world_positions_torch was verified == numpy `_recover_world_positions`
scripts/_oneshot_vae_recon_specific.py:19:    AnyTopDataset, collate_fn as anytop_collate_fn, _recover_world_positions,
scripts/_oneshot_vae_recon_specific.py:139:    pred_world = _recover_world_positions(pred_raw)
src/models/graph_salad/rot6d_fk_recovery.py:4:AnyTop/SALAD `recover_from_bvh_rot_np` (truebones motion_process.py:750) —
src/models/graph_salad/rot6d_fk_recovery.py:39:def _recover_root_R_and_pos(root_13: torch.Tensor):
src/models/graph_salad/rot6d_fk_recovery.py:88:        root_R, root_pos = _recover_root_R_and_pos(m[b:b + 1, :, 0, :])  # [1,T,3,3],[1,T,3]
scripts/_diag_root_double_rot.py:13:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
scripts/_diag_root_double_rot.py:14:from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa
scripts/_diag_root_double_rot.py:29:    ric = _recover_world_positions(raw)
scripts/_diag_root_double_rot.py:30:    fk = recover_from_bvh_rot_np(raw, parents, offsets)
scripts/_diag_root_info.py:12:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa
scripts/_diag_root_info.py:38:world03 = _recover_world_positions(raw)
scripts/_diag_fk_rootdir.py:19:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa
scripts/_diag_fk_rootdir.py:35:world03 = _recover_world_positions(raw)  # [T,J,3]
scripts/_preflight_ric_vs_fk_L2.py:3:For a sample of train+val clips, compute || recover_from_bvh_rot_np(gt) -
scripts/_preflight_ric_vs_fk_L2.py:4:_recover_world_positions(gt) ||, as % of motion bbox diagonal. This is the FK
scripts/_preflight_ric_vs_fk_L2.py:19:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
scripts/_preflight_ric_vs_fk_L2.py:20:from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa
scripts/_preflight_ric_vs_fk_L2.py:50:        ric = _recover_world_positions(raw)                       # [T,J,3]
scripts/_preflight_ric_vs_fk_L2.py:51:        fk = recover_from_bvh_rot_np(raw, parents, offsets)       # [T,J,3]
scripts/_oneshot_t2m_custom_prompt.py:21:    AnyTopDataset, collate_fn as anytop_collate_fn, _recover_world_positions,
scripts/_oneshot_t2m_custom_prompt.py:148:    pred_world = _recover_world_positions(pred_raw)
scripts/animate_anytop13.py:8:    -> _recover_world_positions  (RIFKE -> Cartesian, AnyTop recovery)
scripts/animate_anytop13.py:40:    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
scripts/animate_anytop13.py:42:from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402  rot6d FK path
scripts/animate_anytop13.py:193:                pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets)
scripts/animate_anytop13.py:194:                gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets)
scripts/animate_anytop13.py:202:                gt_ric = _recover_world_positions(gt_raw)
scripts/animate_anytop13.py:203:                pred_ric = _recover_world_positions(pred_raw)
scripts/animate_anytop13.py:204:                pred_fk = recover_from_bvh_rot_np(pred_raw, parents, offsets)
scripts/animate_anytop13.py:207:                pred_world = _recover_world_positions(pred_raw)             # [T, J, 3]
scripts/_render_rot6d_fk_v2_realroot.py:4:Renders, for each long-chain species, GT(red, _recover_world_positions / 0:3
scripts/_render_rot6d_fk_v2_realroot.py:9:non-root chain, because _recover_world_positions rotates channel-0:3 by the
scripts/_render_rot6d_fk_v2_realroot.py:27:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa
scripts/_render_rot6d_fk_v2_realroot.py:72:    world03 = _recover_world_positions(raw)            # GT red (0:3 path)
scripts/_check_gt_fk_units.py:18:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
scripts/_check_gt_fk_units.py:19:from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa
scripts/_check_gt_fk_units.py:33:    ric = _recover_world_positions(raw)                  # [T,J,3]
scripts/_check_gt_fk_units.py:34:    fk = recover_from_bvh_rot_np(raw, parents, offsets)  # [T,J,3]
scripts/_codex_rot6d_render_brief.md:4:用户决定:VAE/diffusion 的可视化 QA 渲染,GT 和 pred **统一改用 rot6d FK 路径**(官方 recover_from_bvh_rot_np:从 channel 3:9 旋转 + 骨骼 offsets + 父链 FK 恢复世界坐标,骨长严格守恒),取代之前的 pos/RIC 路径(channel 0:3 position)。GT 和 pred 用**同一种**恢复 → apples-to-apples 比较。
scripts/_codex_rot6d_render_brief.md:7:- 官方 recover_from_bvh_rot_np 与 recover_from_bvh_ric_np 在原始数据上误差 <1% bbox(root_diff=0.0000),即两路径等价、FK 正确。
scripts/_codex_rot6d_render_brief.md:11:自包含 numpy 复刻官方 `recover_from_bvh_rot_np`(motion_process.py:750)+ 依赖(`recover_root_quat_and_pos_np`:700、Quaternions.__mul__/from_transforms/__neg__/transforms、Animation.positions_global 4x4 matmul 链、rotation_6d_to_matrix_np)。每个算子逐行 verbatim 官方源。**为何自包含**:原 SALAD 库依赖 numpy.core.umath_tests(新 numpy 已删)+ 重 BVH 依赖链,import 不进来。
scripts/_codex_rot6d_render_brief.md:14:- import recover_from_bvh_rot_np
scripts/_codex_rot6d_render_brief.md:16:- 渲染逻辑:render_mode=="rot6d" 时,GT 和 pred 都用 recover_from_bvh_rot_np(从各自 raw 13ch + item 的 rest_offsets/parent_indices 恢复);否则走旧 pos 路径(pred=_recover_world_positions, gt=motion_features[...,:3])。
scripts/_codex_rot6d_render_brief.md:19:1. **rot6d FK 复刻正确性**:anytop_rot6d_fk.py 的算子是否与官方 verbatim 一致(quaternion 乘法/共轭/from_transforms 符号、parent reindex `rot_q[:,p]=hml[:,j]`、root 修正 `-r_rot_quat*rot_q[:,0]`、positions_global 的 4x4 父子 matmul)?有无 off-by-one / 轴序 / dtype 隐患?
scripts/_codex_rot6d_render_brief.md:23:5. dtype/shape:recover_from_bvh_rot_np 返回 float32 [T,J,3],与下游 np.diff/animate 兼容?
scripts/_smoke_world_geometry_terms.py:48:    # motion_features[...,:3] is dataset's _recover_world_positions(raw) (world GT).
scripts/_render_rot6d_official_fk.py:2:recover_from_bvh_rot_np in self-contained numpy (user request 2026-06-01),
scripts/_render_rot6d_official_fk.py:7:  - rotation_6d_to_matrix_np  : utils/rotation_conversions.py:536 (x=norm(a1),
scripts/_render_rot6d_official_fk.py:12:  - recover_root_quat_and_pos_np / recover_from_bvh_ric_np /
scripts/_render_rot6d_official_fk.py:13:    recover_from_bvh_rot_np : motion_process.py:700/738/750 (verbatim)
scripts/_render_rot6d_official_fk.py:19:blue = rot6d FK (official). SELF-CHECK: both share recover_root_quat_and_pos_np
scripts/_render_rot6d_official_fk.py:36:def rotation_6d_to_matrix_np(cont6d):
scripts/_render_rot6d_official_fk.py:133:def recover_root_quat_and_pos_np(data):  # :700
scripts/_render_rot6d_official_fk.py:134:    r_rot_quat = quat_from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))  # [T,4]
scripts/_render_rot6d_official_fk.py:144:    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[..., 0, :])
scripts/_render_rot6d_official_fk.py:154:def recover_from_bvh_rot_np(data, parents, offsets):  # :750 (rot6d FK path)
scripts/_render_rot6d_official_fk.py:155:    r_rot_quat, r_pos = recover_root_quat_and_pos_np(data[:, 0])           # [T,4],[T,3]
scripts/_render_rot6d_official_fk.py:157:    nonroot_mat = rotation_6d_to_matrix_np(data[..., 1:, 3:9])             # [T,J-1,3,3]
scripts/_render_rot6d_official_fk.py:210:    rot = recover_from_bvh_rot_np(raw, parents, offsets).astype(np.float32)  # blue
scripts/render_planet_zoo_clean_qa.py:24:from src.data.anytop_dataset import _recover_world_positions
scripts/render_planet_zoo_clean_qa.py:224:        pos = normalize_ground(_recover_world_positions(motion))
scripts/animate_denoiser.py:46:    _recover_world_positions, _STD_FLOOR,
scripts/animate_denoiser.py:308:        pred_world = _recover_world_positions(pred_raw)
scripts/render_planet_zoo_clean_gifs.py:19:from src.data.anytop_dataset import _recover_world_positions
scripts/render_planet_zoo_clean_gifs.py:205:    positions = _recover_world_positions(motion)
scripts/smoke_anytop.py:60:    recovered from AnyTop's RIFKE encoding via _recover_world_positions. So
scripts/_smoke_fk_fix_torch.py:17:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
scripts/_smoke_fk_fix_torch.py:18:from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa
scripts/_smoke_fk_fix_torch.py:41:    ric = _recover_world_positions(raw.astype(np.float32))
scripts/_smoke_fk_fix_torch.py:42:    fk = recover_from_bvh_rot_np(raw.astype(np.float64), parents, offsets)
scripts/_smoke_fk_fix_torch.py:49:ric = _recover_world_positions(raw.astype(np.float32))
scripts/_smoke_fk_fix_torch.py:50:fk_np = recover_from_bvh_rot_np(raw.astype(np.float64), parents, offsets)
scripts/_diag_vae_nan.py:30:    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
scripts/_diag_vae_nan.py:31:    _rotation_6d_to_matrix_np,
scripts/_diag_vae_nan.py:110:                Rm = _rotation_6d_to_matrix_np(rot6d_root)   # [T,3,3]
scripts/_diag_vae_nan.py:118:                world = _recover_world_positions(pred_raw)   # [T,J,3]
scripts/_codex_world_rot6d_fk_brief.md:4:按 handoff/20260601_rot6d_fk_loss_plan.md 实现组合几何监督: L_total = L_anytop13_base + w_world*L_world_ric + w_fk*L_rot6d_fk + w_traj*L_root_traj。保留现有 anytop13_world_geometry(world-only ablation)不删。FK 路线用官方 recover_from_bvh_rot_np(matrix port)对非 root 旋转有梯度,这是与 world_geometry 的核心区别。
scripts/_codex_world_rot6d_fk_brief.md:12:1. **新模块 src/models/graph_salad/rot6d_fk_recovery.py**: recover_rot6d_fk_positions_torch([B,T,J,13]raw, parent_indices, rest_offsets[B,J,3], joint_mask[B,J])->[B,T,J,3]。matrix-only torch port 官方 recover_from_bvh_rot_np:root R+pos 复用 RIC root 路径(world_recovery._rot6d_to_matrix_torch + vel cumsum + height); 非 root R from ch3:9; parent reindex local_R[p]=all_R[j](矩阵直接做,不走 numpy 的 quat round-trip); root 修正 local_R[0]=root_R^T@local_R[0](=numpy 的 -r_rot_quat*rot_q[0]); 4x4 FK 链(list 累积避免 autograd inplace)。
scripts/_codex_world_rot6d_fk_brief.md:18:1. **FK 正确性**: rot6d_fk_recovery.py 的 matrix-only 版是否与官方 recover_from_bvh_rot_np 逻辑等价? 重点: parent reindex(local_R[p]=all_R[j] 而非 [j]=[p])、root 修正 root_R^T@local_R[0]、root pos/R 复用 RIC 路径是否一致? smoke 已证 vs numpy 1.19e-6,但请核对逻辑无隐藏 off-by-one/排列错。autograd: list 累积 + out.clone() 是否真无 inplace 破图(smoke backward 成功)?
scripts/_render_vae_recon_rot6d_vs_pos.py:4:  BLUE = rot6d FK path  : recover_from_bvh_rot_np(pred_13ch, parents, offsets)
scripts/_render_vae_recon_rot6d_vs_pos.py:107:    rqj[:, 0] = qm(qn(rq), rqj[:, 0])
scripts/_verify_official_rot_vs_ric_rawdata.py:2:recover_from_bvh_ric_np vs recover_from_bvh_rot_np, motion + cond from the SAME
scripts/_verify_official_rot_vs_ric_rawdata.py:30:def rotation_6d_to_matrix_np(c):
scripts/_verify_official_rot_vs_ric_rawdata.py:93:    rq = quat_from_transforms(rotation_6d_to_matrix_np(data[:, 3:9]))
scripts/_verify_official_rot_vs_ric_rawdata.py:109:    rmat = qmat(rq); nrm = rotation_6d_to_matrix_np(data[..., 1:, 3:9])
scripts/_verify_official_rot_vs_ric_rawdata.py:114:    rqj[:, 0] = qmul(qneg(rq), rqj[:, 0])
scripts/_codex_fk_fix_brief.md:12:RIC 路(从 ch0:3 root-invariant position 恢复, `_recover_world_positions`)= 可靠
scripts/_codex_fk_fix_brief.md:25:token **已含全局朝向变化**;reindex 后 rot_q[:,0] 已是正确 root 朝向,再乘
scripts/_codex_fk_fix_brief.md:29:1. **src/data/anytop_rot6d_fk.py** (numpy, `recover_from_bvh_rot_np`):
scripts/_codex_fk_fix_brief.md:30:   删 `rot_q[:, 0] = _quat_mul(_quat_neg(r_rot_quat), rot_q[:, 0])`,替为注释。
scripts/_codex_fk_fix_brief.md:55:   torch 的 `root_R`(仍用于 `all_R` concat 占位 + `_recover_root_R_and_pos` 的 root
scripts/_codex_fk_fix_brief.md:69:- src/data/anytop_dataset.py:282 `_recover_world_positions` (RIC ground truth, 对照)
scripts/_render_gt_ric_vs_fk.py:20:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
scripts/_render_gt_ric_vs_fk.py:21:from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa
scripts/_render_gt_ric_vs_fk.py:41:    ric = _recover_world_positions(raw)
scripts/_render_gt_ric_vs_fk.py:42:    fk = recover_from_bvh_rot_np(raw, parents, offsets)
scripts/_smoke_rot6d_fk_torch.py:12:from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa numpy ref
scripts/_smoke_rot6d_fk_torch.py:36:    fk_np = recover_from_bvh_rot_np(raw, parents, offsets)            # [T,J,3]
scripts/_render_rot6d_fk_sanity.py:3:  RED  (ref) = _recover_world_positions — the CURRENT renderer, which uses
scripts/_render_rot6d_fk_sanity.py:28:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions  # noqa: E402
scripts/_render_rot6d_fk_sanity.py:58:    world03 = _recover_world_positions(raw)                # [T,J,3]
scripts/_smoke_world_recovery_torch.py:4:  - numpy/scipy  src.data.anytop_dataset._recover_world_positions  (per [T,J,13])
scripts/_smoke_world_recovery_torch.py:20:    AnyTopDataset, _recover_world_positions,
scripts/_smoke_world_recovery_torch.py:72:        world_np = _recover_world_positions(raw_tjc.astype(np.float32))  # [T,J,3]
scripts/_diag_fk_variants.py:2:against the RELIABLE ground truth = RIC path (_recover_world_positions).
scripts/_diag_fk_variants.py:13:from src.data.anytop_dataset import AnyTopDataset, _recover_world_positions, _STD_FLOOR  # noqa
scripts/_diag_fk_variants.py:15:    _recover_root_quat_and_pos_np, _rotation_6d_to_matrix_np, _quat_from_transforms,
scripts/_diag_fk_variants.py:16:    _quat_to_matrix, _quat_mul, _quat_neg, _positions_global, recover_from_bvh_rot_np)
scripts/_diag_fk_variants.py:37:    ric = _recover_world_positions(raw.astype(np.float32))
scripts/_diag_fk_variants.py:38:    fk = recover_from_bvh_rot_np(raw, parents, offsets)
scripts/_diag_fk_variants.py:45:r_rot_quat, r_pos = _recover_root_quat_and_pos_np(raw[:, 0])
scripts/_diag_fk_variants.py:47:nonroot_mat = _rotation_6d_to_matrix_np(raw[:, 1:, 3:9])
scripts/_diag_fk_variants.py:79:rq = reindex(); rq[:, 0] = _quat_mul(_quat_neg(r_rot_quat), rq[:, 0]); run(rq, "A reindex+corr(CURRENT)")
scripts/_diag_fk_variants.py:81:rq = all_q.copy(); rq[:, 0] = _quat_mul(_quat_neg(r_rot_quat), rq[:, 0]); run(rq, "C self-slot+corr")

codex
There are older scripts in the repo that still contain the removed double-root-rotation, but the scoped large-render script only shows it in a comment so far. I’m now checking canonical RIC recovery and the checkpoint/model path, then I’ll run small equivalence commands on real samples.
exec
/usr/bin/bash -lc "nl -ba src/data/anytop_dataset.py | sed -n '70,90p;250,350p;805,835p;985,1035p;1040,1065p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
exec
/usr/bin/bash -lc "nl -ba scripts/animate_anytop13.py | sed -n '1,80p;150,210p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
     1	"""Visual QA for the M1.7 AnyTop-native 13ch Graph-VAE (feat_mode=anytop13).
     2	
     3	The anytop13 decoder outputs `pred_motion [B,T,J,13]` in AnyTop's NORMALIZED
     4	RIFKE space — not directly plottable. This tool closes the loop:
     5	
     6	  pred_motion (normalized 13ch)
     7	    -> de-normalize  (× std + mean, AnyTop's per-joint stats)
     8	    -> _recover_world_positions  (RIFKE -> Cartesian, AnyTop recovery)
     9	    -> animate_clip  -> GT-vs-pred GIF
    10	
    11	GT world positions are taken directly from `batch.motion_features[...,:3]` —
    12	the dataset already recovered them from the raw clip, so GT needs no de-norm.
    13	
    14	Per cross-project rule "可视化 demo 准确度 > metric": this is THE truth gate
    15	for anytop13 — recon loss numbers must be cross-checked against visible motion
    16	fidelity (frozen / jitter / collapse are invisible in metrics, visible here).
    17	
    18	Usage:
    19	  python scripts/animate_anytop13.py \\
    20	      --ckpt runs/m1_7_anytop13_dynamic_seed42/best_recon_model.pt \\
    21	      --species Alligator,Spider,Trex,Dragon --n_per 2 \\
    22	      --out runs/m1_7_anytop13_dynamic_seed42/qa_animate
    23	"""
    24	
    25	import argparse
    26	import sys
    27	from pathlib import Path
    28	
    29	import matplotlib
    30	import numpy as np
    31	import torch
    32	
    33	matplotlib.use("Agg")
    34	
    35	project_root = Path(__file__).resolve().parents[1]
    36	sys.path.insert(0, str(project_root))
    37	
    38	from scripts.animate import animate_clip, animate_clip_3col, contact_sheet  # noqa: E402  model-agnostic renderers
    39	from src.data.anytop_dataset import (  # noqa: E402
    40	    AnyTopDataset, collate_fn, _recover_world_positions, _STD_FLOOR,
    41	)
    42	from src.data.anytop_rot6d_fk import recover_from_bvh_rot_np  # noqa: E402  rot6d FK path
    43	from src.models.graph_salad import GraphMotionBatch, GraphMotionVAE  # noqa: E402
    44	
    45	
    46	def load_anytop13_vae(ckpt_path: str, dev: torch.device):
    47	    """Reconstruct an anytop13 GraphMotionVAE from a train_graph_vae.py ckpt.
    48	
    49	    Uses the ckpt's saved `args` to rebuild the exact model, then loads weights
    50	    strict=True. attn_mode / use_text default safely for ckpts saved before
    51	    those flags existed.
    52	    """
    53	    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    54	    ta = ck.get("args", {})
    55	    if not ta:
    56	        raise SystemExit(f"animate_anytop13: ckpt {ckpt_path} missing 'args' key")
    57	    if ta.get("feat_mode") != "anytop13":
    58	        raise SystemExit(
    59	            f"animate_anytop13: ckpt feat_mode={ta.get('feat_mode')!r}, expected "
    60	            f"'anytop13' — use scripts/animate_graph_vae.py for fk6 ckpts"
    61	        )
    62	    vae = GraphMotionVAE(
    63	        pool_type=ta["pool_type"],
    64	        pool_tau=ta.get("pool_tau"),
    65	        d_model=ta["d_model"], n_heads=ta["n_heads"], d_ff=ta["d_ff"],
    66	        n_graph_layers=ta["n_graph_layers"],
    67	        n_enc_temporal_layers=ta["n_enc_temporal_layers"],
    68	        n_cross_layers=ta["n_cross_layers"],
    69	        n_dec_temporal_layers=ta["n_dec_temporal_layers"],
    70	        n_treeik_layers=ta["n_treeik_layers"],
    71	        max_coarse=ta["max_coarse"], local_radius=ta["local_radius"],
    72	        temporal_stride=ta["temporal_stride"],
    73	        temporal_kernel=ta["temporal_kernel"],
    74	        dropout=ta["dropout"],
    75	        feat_mode="anytop13",
    76	        attn_mode=ta.get("attn_mode") or "scalar",
    77	        use_text=bool(ta.get("use_text") or False),
    78	        decoder_mode=ta.get("decoder_mode") or "unpool_identity",
    79	        n_graph_temporal_layers=ta.get("n_graph_temporal_layers", 4),
    80	    ).to(dev)
   150	    summary = []
   151	
   152	    with torch.no_grad():
   153	        for i in range(len(ds)):
   154	            item = ds[i]
   155	            sp = item["object_type"]
   156	            if sp not in picked or picked[sp] >= args.n_per:
   157	                continue
   158	            raw = collate_fn([item])
   159	            raw = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in raw.items()}
   160	            batch = GraphMotionBatch.from_collate_dict(raw)
   161	            out = vae(batch, sample=False)               # deterministic z=mu
   162	
   163	            J = int(item["num_joints"])
   164	            # Stride-aware T (2026-05-28, mirrors animate_denoiser.py codex P1
   165	            # fix 2026-05-23): VAE temporal_stride=4 collapses every 4 frames
   166	            # via .all() on encode, so a clip with T_var=67 has only
   167	            # floor(67/4)*4=64 valid latent-recovered frames; the trailing 3
   168	            # frames come back zero/garbage from decoder. Use frame_mask
   169	            # _recovered to clip — otherwise GIF tail shows collapse/jitter
   170	            # that LOOKS like recon failure but is just stride-incomplete tail.
   171	            T_clip = int(item["num_frames"])
   172	            T_valid = int(out["frame_mask_recovered"][0].sum().item())
   173	            T = min(T_clip, T_valid)
   174	            T_dropped = T_clip - T
   175	            # De-normalize pred 13ch: raw = norm * (std + eps) + mean.
   176	            # anytop_mean/std are de-norm stats — only needed for visualization,
   177	            # not a model input, so they ride the raw collate dict (not the
   178	            # typed GraphMotionBatch).
   179	            std = raw["anytop_std"][0, :J].cpu().numpy()    # [J, 13]
   180	            mean = raw["anytop_mean"][0, :J].cpu().numpy()  # [J, 13]
   181	            pred_norm = out["pred_motion"][0, :T, :J, :].cpu().numpy()  # [T, J, 13]
   182	            pred_raw = pred_norm * (std[None] + _STD_FLOOR) + mean[None]
   183	            parents = [int(p) for p in item["parent_indices"][:J]]
   184	            # Render mode: rot6d FK (default) or legacy pos/RIC. BOTH GT and pred
   185	            # use the SAME recovery so the visual comparison is apples-to-apples.
   186	            if args.render_mode == "rot6d":
   187	                # GT raw 13ch (same de-norm as pred); offsets share the item's
   188	                # new_to_old_perm ordering with anytop_x, so FK is aligned.
   189	                # anytop_x is a [J,13,T] tensor -> numpy [T,J,13].
   190	                gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :]
   191	                gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]
   192	                offsets = np.asarray(item["rest_offsets"])[:J]
   193	                pred_world = recover_from_bvh_rot_np(pred_raw, parents, offsets)
   194	                gt_world = recover_from_bvh_rot_np(gt_raw, parents, offsets)
   195	            elif args.render_mode == "three_col":
   196	                # 3-col QA: GT_RIC | PRED_RIC | PRED_FK. pred goes through BOTH the
   197	                # RIC/position route AND the rot6d-FK route; their agreement is the
   198	                # core rot6d_fk signal. speed_ratio reported on the RIC route.
   199	                gt_norm = np.asarray(item["anytop_x"]).transpose(2, 0, 1)[:T, :J, :]
   200	                gt_raw = gt_norm * (std[None] + _STD_FLOOR) + mean[None]
   201	                offsets = np.asarray(item["rest_offsets"])[:J]
   202	                gt_ric = _recover_world_positions(gt_raw)
   203	                pred_ric = _recover_world_positions(pred_raw)
   204	                pred_fk = recover_from_bvh_rot_np(pred_raw, parents, offsets)
   205	                pred_world, gt_world = pred_ric, gt_ric
   206	            else:  # pos (legacy RIC path)
   207	                pred_world = _recover_world_positions(pred_raw)             # [T, J, 3]
   208	                gt_world = batch.motion_features[0, :T, :J, :3].cpu().numpy()
   209	
   210	            k = picked[sp]

 succeeded in 0ms:
    70	# motion_texts_by_file.json), copied into this project to decouple training
    71	# from the external AnyTop repo path. The AnyTop source is read-only and
    72	# never modified; this is an independent copy. Override with `data_root` /
    73	# `--anytop_root` / `ANYTOP_ROOT` to point elsewhere (e.g. the AnyTop repo
    74	# original at .../Anytop/AnyTop/dataset/truebones/zoo/truebones_processed).
    75	_DEFAULT_ANYTOP_ROOT = (
    76	    "/iridisfs/scratch/ts1v23/workspace/noKslot_clean/data/anytop_truebones"
    77	)
    78	_STD_FLOOR = 1e-6  # matches AnyTop's `std += 1e-6` stability constant
    79	
    80	
    81	def _longest_prefix_match(fname: str, keys_sorted_desc: list[str]) -> Optional[str]:
    82	    """Match a filename to its cond object_type by longest-prefix.
    83	
    84	    AnyTop ships motions in two naming conventions:
    85	      "Alligator___BigMouth_5.npy"     -> object_type "Alligator"
    86	      "Cat_CAT_IdlePurr_195.npy"       -> object_type "Cat"
    87	      "Fox_-_Attack1_361.npy"          -> object_type "Fox"
    88	    so a plain `split("___")` misses 45 / 1070 files. `keys_sorted_desc` is
    89	    cond.keys() sorted by len(key) descending so a "BrownBear" file resolves
    90	    before a "Bear" prefix match would be tried.
   250	    removal_options = [j for j in ee if j not in feet]
   251	    n_remove = int(np.floor(len(removal_options) * removal_rate))
   252	    if n_remove <= 0:
   253	        return raw_motion, sk
   254	    remove = sorted(rng.sample(removal_options, n_remove), reverse=True)
   255	
   256	    new_motion = np.delete(raw_motion, remove, axis=1)
   257	    new_parents = np.delete(parents, remove, axis=0)
   258	    # Decrement parent pointers above each removed index (descending order so
   259	    # each decrement sees indices consistent with the prior step).
   260	    for rj in remove:
   261	        new_parents[new_parents > rj] -= 1
   262	    new_offsets = np.delete(np.asarray(sk["offsets"], dtype=np.float32), remove, axis=0)
   263	    new_tpos = np.delete(np.asarray(sk["tpos_first_frame"], dtype=np.float32), remove, axis=0)
   264	    new_mean = np.delete(np.asarray(sk["mean"], dtype=np.float32), remove, axis=0)
   265	    new_std = np.delete(np.asarray(sk["std"], dtype=np.float32), remove, axis=0)
   266	    new_names = [n for k, n in enumerate(sk["joint_names"]) if k not in set(remove)]
   267	
   268	    derived = _build_derived(new_parents, new_offsets, new_names)
   269	    reduced_sk = {
   270	        "n_joints": len(new_parents),
   271	        "parents": new_parents,
   272	        "joint_names": new_names,
   273	        "offsets": new_offsets,
   274	        "tpos_first_frame": new_tpos,
   275	        "mean": new_mean,
   276	        "std": new_std,
   277	        **derived,
   278	    }
   279	    return new_motion, reduced_sk
   280	
   281	
   282	def _recover_world_positions(motion_13ch: np.ndarray) -> np.ndarray:
   283	    """Recover world-space [T, J, 3] joint positions from AnyTop RIFKE encoding.
   284	
   285	    Mirrors AnyTop motion_process.recover_from_bvh_ric_np (line 493):
   286	      1. Root rotation per frame from 6D rot at channels 3:9.
   287	      2. Root xz position via cumulative sum of velocities at channels 9 & 11,
   288	         applied AFTER inverse-rotating the per-frame velocity into the world
   289	         frame (so cumsum acts in world space).
   290	      3. Root y position from channel 1 (height stored directly, not integrated).
   291	      4. Non-root joint positions: channels 0:3 are root-relative; rotate them
   292	         by inverse root rotation per frame to go to world frame, then add
   293	         root xz.
   294	
   295	    Args:
   296	      motion_13ch: [T, J, 13] raw (un-normalized) AnyTop motion encoding.
   297	    Returns:
   298	      [T, J, 3] world-space joint positions.
   299	    """
   300	    if motion_13ch.ndim != 3 or motion_13ch.shape[-1] != 13:
   301	        raise ValueError(
   302	            f"motion_13ch must be [T, J, 13], got {motion_13ch.shape}"
   303	        )
   304	    motion = motion_13ch.astype(np.float32)
   305	    T, J, _ = motion.shape
   306	    root = motion[:, 0, :]  # [T, 13]
   307	
   308	    # 1. Root rotation per frame from 6D rot (channels 3:9).
   309	    rot_mat = _rotation_6d_to_matrix_np(root[:, 3:9])  # [T, 3, 3]
   310	    root_rot = _ScipyRotation.from_matrix(rot_mat)     # [T]
   311	
   312	    # 2. Root xz integration: shift-by-1 vel (no motion at t=0), inverse-rotate
   313	    #    per frame, cumsum. AnyTop's code uses indices 9 (x) and 11 (z); idx 10
   314	    #    is NOT used in root recovery (it's per-joint vel_y elsewhere).
   315	    rpos_local = np.zeros((T, 3), dtype=np.float32)
   316	    rpos_local[1:, 0] = root[:-1, 9]   # vel_x at t-1
   317	    rpos_local[1:, 2] = root[:-1, 11]  # vel_z at t-1
   318	    # Apply inverse rotation per frame (no broadcasting in scipy; loop is cheap).
   319	    inv_rot = root_rot.inv()
   320	    rpos_world = np.zeros_like(rpos_local)
   321	    for t in range(T):
   322	        rpos_world[t] = inv_rot[t].apply(rpos_local[t])
   323	    rpos_world = np.cumsum(rpos_world, axis=0)
   324	    rpos_world[:, 1] = root[:, 1]  # root height directly from channel 1
   325	
   326	    # 3. Non-root joints: rotate root-relative pos (channels 0:3) to world.
   327	    if J > 1:
   328	        rel = motion[:, 1:, :3].astype(np.float32)  # [T, J-1, 3]
   329	        world_rel = np.zeros_like(rel)
   330	        for t in range(T):
   331	            world_rel[t] = inv_rot[t].apply(rel[t])  # [J-1, 3]
   332	        # Add root xz (NOT root y — AnyTop encodes root y directly per frame
   333	        # at root.channel_1; non-root joints carry their own y as part of
   334	        # root-relative pos channels 0:3 -> after inverse-rotate, they're in
   335	        # world frame already except for the missing root xz origin shift).
   336	        world_rel[..., 0] += rpos_world[:, None, 0]
   337	        world_rel[..., 2] += rpos_world[:, None, 2]
   338	    else:
   339	        world_rel = np.zeros((T, 0, 3), dtype=np.float32)
   340	
   341	    # Concatenate root world pos at index 0
   342	    world_positions = np.concatenate(
   343	        [rpos_world[:, None, :], world_rel], axis=1
   344	    )  # [T, J, 3]
   345	    return world_positions.astype(np.float32)
   346	
   347	
   348	def _parents_to_adjacency(parents: np.ndarray, J: int) -> np.ndarray:
   349	    """Symmetric binary adjacency from parent_indices. Self-loops excluded."""
   350	    A = np.zeros((J, J), dtype=np.float32)
   805	        # `_remove_joints_aug` never mutates `c` (works on local numpy copies).
   806	        if self.augment and random.random() < self.augment_prob:
   807	            raw_motion, sk = _remove_joints_aug(
   808	                raw_motion, c, self.removal_rate, random.Random()
   809	            )
   810	        else:
   811	            sk = c
   812	        J_orig = sk["n_joints"]
   813	
   814	        T_var = raw_motion.shape[0]
   815	
   816	        # ---------- AnyTop normalized 13ch view (for the future end-to-end path) ----------
   817	        mean = sk["mean"]               # [J_orig, 13] RAW (pre-normalize)
   818	        std = sk["std"]                 # [J_orig, 13]
   819	        std_safe = std + _STD_FLOOR
   820	        normed_13 = (raw_motion - mean[None, :, :]) / std_safe[None, :, :]
   821	        normed_13 = np.nan_to_num(normed_13).astype(np.float32)
   822	        # tpos normalized for AnyTop extra key parity.
   823	        tpos_norm = np.nan_to_num(
   824	            ((sk["tpos_first_frame"] - mean) / std_safe).astype(np.float32)
   825	        )
   826	
   827	        # ---------- 6ch view: WORLD positions via AnyTop recovery (codex P1 #2) ----------
   828	        # Recover from RAW 13ch (NOT normalized — AnyTop's recover assumes raw).
   829	        world_pos = _recover_world_positions(raw_motion)        # [T_var, J_orig, 3]
   830	        # World velocity: numerical diff × fps, zero-pad at t=0.
   831	        world_vel = np.zeros_like(world_pos)
   832	        if T_var >= 2:
   833	            world_vel[1:] = (world_pos[1:] - world_pos[:-1]) * self.target_fps
   834	            world_vel[0] = world_vel[1]
   835	        # Stack into 6ch view in FK-ordered J axis.
   985	            else:
   986	                caption = self.captions.get(info["motion_id"], "")
   987	        else:
   988	            caption_emb = np.zeros(self._caption_emb_dim, dtype=np.float32)
   989	            has_text = False
   990	            caption = self.captions.get(info["motion_id"], "")
   991	
   992	        return {
   993	            # ---- GraphMotionBatch-compatible padded tensors ----
   994	            "motion_features": torch.from_numpy(motion_6ch),               # [T, Jm, 6]
   995	            "skeleton_features": torch.from_numpy(skel_feats_padded),      # [Jm, 9]
   996	            "joint_mask": torch.from_numpy(joint_mask),                    # [Jm] bool
   997	            "frame_mask": torch.from_numpy(frame_mask),                    # [Tm] bool
   998	            "adjacency": torch.from_numpy(adjacency_padded),               # [Jm, Jm]
   999	            "geodesic_dist": torch.from_numpy(geo_padded),                 # [Jm, Jm]
  1000	            "name_hashes": torch.from_numpy(name_hashes_padded),           # [Jm] int64
  1001	            "root_position": torch.from_numpy(root_pos_padded),            # [Tm, 3]
  1002	            "root_velocity": torch.from_numpy(root_vel_padded),            # [Tm, 3]
  1003	            "local_rotations_6d": torch.from_numpy(rot6d_padded),          # [Tm, Jm, 6]
  1004	            "foot_contact": torch.from_numpy(contact_padded),              # [Tm, 4]
  1005	            "bone_lengths": torch.from_numpy(bone_padded),                 # [Tm, Jm]
  1006	            "rest_offsets": torch.from_numpy(rest_offsets_padded),         # [Jm, 3]
  1007	
  1008	            # ---- Batched scalars (collate -> [B] tensors) ----
  1009	            "num_joints": int(J_orig),
  1010	            "num_frames": int(actual_T),
  1011	            "fps": float(self.target_fps),
  1012	            "has_rotations": True,  # AnyTop always carries 6D rot in channels 3:9
  1013	
  1014	            # ---- Per-sample lists ----
  1015	            "parent_indices": parent_indices_list,
  1016	            "joint_names": joint_names_list,
  1017	            "canonical_names": canonical_names_list,
  1018	            "bone_lengths_rest": bone_lengths_rest_list,
  1019	
  1020	            # ---- Per-sample strings ----
  1021	            "text": caption,
  1022	            "skeleton_id": info["object_type"],
  1023	            "motion_id": info["motion_id"],
  1024	
  1025	            # ---- AnyTop-specific extras (NOT validated by GraphMotionBatch) ----
  1026	            "anytop_x": torch.from_numpy(anytop_x),                        # [Jm, 13, Tm] NORMALIZED
  1027	            "anytop_graph_dist": torch.from_numpy(anytop_gd_padded),        # [Jm, Jm] (AnyTop clamped ≤5)
  1028	            "anytop_joint_relations": torch.from_numpy(jrel_padded),       # [Jm, Jm]
  1029	            "anytop_tpos_first_frame": torch.from_numpy(tpos_padded),      # [Jm, 13]
  1030	            "anytop_mean": torch.from_numpy(mean_padded),                  # [Jm, 13]
  1031	            "anytop_std": torch.from_numpy(std_padded),                    # [Jm, 13]
  1032	            # codex P1 #8: per-joint contact (AnyTop channel-12 source of truth)
  1033	            "foot_contact_per_joint": torch.from_numpy(contact_per_joint_padded),  # [Tm, Jm]
  1034	            # M1.7 Task 2: optional text condition (precomputed T5 caption embedding)
  1035	            "caption_emb": torch.from_numpy(caption_emb),                  # [768] f32
  1040	
  1041	
  1042	def collate_fn(batch: list[dict]) -> dict:
  1043	    """Stack tensors, list everything else.
  1044	
  1045	    Compatible with src/data/unified_dataset.py:collate_fn semantics so the
  1046	    resulting dict feeds GraphMotionBatch.from_collate_dict() unchanged.
  1047	    """
  1048	    result: dict = {}
  1049	    keys = batch[0].keys()
  1050	    for key in keys:
  1051	        v0 = batch[0][key]
  1052	        if key == "text":
  1053	            result[key] = [b[key] for b in batch]
  1054	        elif isinstance(v0, torch.Tensor):
  1055	            result[key] = torch.stack([b[key] for b in batch])
  1056	        elif isinstance(v0, bool):
  1057	            # bool ⊂ int in Python — must branch first or collate emits int64 ones.
  1058	            result[key] = torch.tensor([b[key] for b in batch], dtype=torch.bool)
  1059	        elif isinstance(v0, (int, float)):
  1060	            # Stay aligned with UnifiedMotionDataset.collate_fn: ints -> int64,
  1061	            # floats -> float32 via torch.tensor default rules.
  1062	            if isinstance(v0, int):
  1063	                result[key] = torch.tensor([b[key] for b in batch], dtype=torch.int64)
  1064	            else:
  1065	                result[key] = torch.tensor([b[key] for b in batch], dtype=torch.float32)

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/batch.py | sed -n '141,160p;517,545p;590,600p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
   141	    def from_collate_dict(cls, d: dict[str, Any]) -> "GraphMotionBatch":
   142	        """Construct from `collate_fn` output, fail loud on schema drift.
   143	
   144	        Validates (in order):
   145	          1. All required keys present.
   146	          2. Padded tensors: type, rank, last-dim, dtype (per ``_TENSOR_SHAPE_SPEC``).
   147	          3. Square check for adjacency / geodesic_dist.
   148	          4. Batched scalars: type, shape == [B], dtype (per ``_BATCHED_SCALAR_SPEC``).
   149	          5. Per-sample lists: type list, length == B.
   150	          6. Cross-tensor T_max + J_max consistency.
   151	
   152	        Raises ValueError with offending key on first mismatch.
   153	        """
   154	        # --- 1. Required-key presence ---
   155	        all_required_keys = (
   156	            [k for k, *_ in _TENSOR_SHAPE_SPEC]
   157	            + [k for k, *_ in _BATCHED_SCALAR_SPEC]
   158	            + list(_PER_SAMPLE_LIST_KEYS)
   159	            + list(_PER_SAMPLE_STRING_KEYS)
   160	        )
   517	            ("anytop_mean",            3, (J_max_val, 13)),
   518	            ("anytop_std",             3, (J_max_val, 13)),
   519	        )
   520	        for key, expected_rank, tail_shape in _OPTIONAL_TENSOR_SPEC:
   521	            if key not in d:
   522	                continue
   523	            t = d[key]
   524	            if not isinstance(t, torch.Tensor):
   525	                raise ValueError(
   526	                    f"GraphMotionBatch: optional '{key}' must be torch.Tensor, "
   527	                    f"got {type(t).__name__}"
   528	                )
   529	            if t.dim() != expected_rank or tuple(t.shape) != (B, *tail_shape):
   530	                raise ValueError(
   531	                    f"GraphMotionBatch: optional '{key}' must have shape "
   532	                    f"{(B, *tail_shape)}, got {tuple(t.shape)}"
   533	                )
   534	            if t.dtype != torch.float32:
   535	                raise ValueError(
   536	                    f"GraphMotionBatch: optional '{key}' dtype must be float32, "
   537	                    f"got {t.dtype}"
   538	                )
   539	            if t.device != ref_device:
   540	                raise ValueError(
   541	                    f"GraphMotionBatch: optional '{key}' device {t.device} != "
   542	                    f"motion_features device {ref_device}"
   543	                )
   544	            if not torch.isfinite(t).all():
   545	                raise ValueError(
   590	            text=d["text"],
   591	            skeleton_id=d["skeleton_id"],
   592	            motion_id=d["motion_id"],
   593	            anytop_x=d.get("anytop_x"),
   594	            anytop_graph_dist=d.get("anytop_graph_dist"),
   595	            anytop_joint_relations=d.get("anytop_joint_relations"),
   596	            foot_contact_per_joint=d.get("foot_contact_per_joint"),
   597	            anytop_mean=d.get("anytop_mean"),
   598	            anytop_std=d.get("anytop_std"),
   599	            caption_emb=d.get("caption_emb"),
   600	            has_text=d.get("has_text"),

exec
/usr/bin/bash -lc "nl -ba src/models/graph_salad/vae.py | sed -n '600,725p;746,760p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
   600	    def decode(self, encode_out: dict, batch: "GraphMotionBatch") -> dict:
   601	        """Unpool → decoder head.
   602	
   603	        fk6:      → TreeIK FK → pred_pos/pred_vel [B, T, J, 3].
   604	        anytop13: → direct 13ch regression → pred_motion [B, T, J, 13].
   605	        The return dict always carries pred_pos/pred_vel/pred_motion; the
   606	        keys not produced by the active feat_mode are None.
   607	        """
   608	        z = encode_out["z"]
   609	        s_j = encode_out["s_j"]
   610	        assignment = encode_out["assignment"]
   611	        coarse_mask = encode_out["coarse_mask"]
   612	        frame_mask_lat = encode_out["frame_mask_lat"]
   613	
   614	        # ---- Optional text conditioning ----
   615	        # Project the T5 caption embedding once; each decoder_mode branch
   616	        # broadcasts it into whichever tensor feeds the decoder. Gated by
   617	        # has_text so a missing-caption sample contributes nothing (a zero
   618	        # embedding still picks up text_proj's bias, hence the explicit gate).
   619	        text_vec = None
   620	        if self.use_text:
   621	            if batch.caption_emb is None or batch.has_text is None:
   622	                raise ValueError(
   623	                    "GraphMotionVAE(use_text=True) requires batch.caption_emb "
   624	                    "and batch.has_text"
   625	                )
   626	            text_vec = self.text_proj(batch.caption_emb)            # [B, D]
   627	            text_vec = text_vec * batch.has_text[:, None].to(text_vec.dtype)
   628	
   629	        if self.decoder_mode in ("coarse_xattn", "graph_temporal"):
   630	            # Pass the coarse slots + the REAL pool assignment P straight to
   631	            # MotionDecoder. Its step-1 einsum does the P-weighted unpool, and
   632	            # — crucially — the cross-attention layers KEEP using P (bias =
   633	            # log P), so each fine joint attends to its coarse anchors instead
   634	            # of being locked to itself by an identity assignment.
   635	            # DynamicGraphUnpool is not used on this path.
   636	            # graph_temporal enters here too, then refines `feats` with the
   637	            # extra spatial+temporal layers (see the block after the if/else).
   638	            z_up = z.repeat_interleave(self.temporal_stride, dim=1)   # [B,T,C,D]
   639	            frame_mask_recovered = frame_mask_lat.repeat_interleave(
   640	                self.temporal_stride, dim=-1)
   641	            if text_vec is not None:
   642	                z_up = z_up + text_vec[:, None, None, :]              # broadcast (T,C)
   643	            feats = self.decoder(
   644	                z_up,                  # slot_features [B, T, C, D]
   645	                s_j,                   # skeleton embeddings [B, J, D]
   646	                assignment,            # REAL pool assignment P [B, J, C]
   647	                batch.joint_mask,
   648	                frame_mask_recovered,
   649	                return_features=True,
   650	            )  # [B, T, J, D]
   651	        else:  # "unpool_identity" — legacy: unpool to fine joints, identity decoder
   652	            if self.unpool is not None:
   653	                unpool_out = self.unpool(
   654	                    coarse_features=z,
   655	                    assignment=assignment,
   656	                    joint_mask=batch.joint_mask,
   657	                    coarse_mask=coarse_mask,
   658	                    frame_mask_down=frame_mask_lat,
   659	                )
   660	                h_fine = unpool_out["fine_features"]   # [B, T, J, D]
   661	                frame_mask_recovered = unpool_out["frame_mask_up"]
   662	            else:
   663	                # No-pool: temporal upsample only
   664	                B, T_lat, J, D = z.shape
   665	                T_full = T_lat * self.temporal_stride
   666	                z_flat = z.permute(0, 2, 3, 1).reshape(B * J * D, 1, T_lat)
   667	                h_up = z_flat.repeat_interleave(self.temporal_stride, dim=-1)
   668	                h_fine = h_up.reshape(B, J, D, T_full).permute(0, 3, 1, 2).contiguous()
   669	                frame_mask_recovered = frame_mask_lat.repeat_interleave(
   670	                    self.temporal_stride, dim=-1)
   671	            if text_vec is not None:
   672	                h_fine = h_fine + text_vec[:, None, None, :]          # broadcast (T,J)
   673	            # MotionDecoder with an IDENTITY assignment (K == J): the cross-
   674	            # attention degenerates to per-joint self-refinement.
   675	            asg = _identity_assignment(batch.joint_mask)  # [B, J, J]
   676	            feats = self.decoder(
   677	                h_fine,                # slot_features [B, T, K=J, D]
   678	                s_j,                   # skeleton embeddings [B, J, D]
   679	                asg,                   # identity assignment [B, J, K=J]
   680	                batch.joint_mask,
   681	                frame_mask_recovered,
   682	                return_features=True,
   683	            )  # [B, T, J, D]
   684	
   685	        if self.decoder_mode == "graph_temporal":
   686	            # Refine the fine features with AnyTop-style spatial(graph)+temporal
   687	            # layers — the joint↔joint + long-range temporal coordination
   688	            # MotionDecoder lacks. Each layer re-masks padded joints/frames so
   689	            # `feats` stays clean by construction.
   690	            if batch.anytop_graph_dist is None or batch.anytop_joint_relations is None:
   691	                raise ValueError(
   692	                    "decoder_mode='graph_temporal' requires batch.anytop_graph_dist "
   693	                    "and batch.anytop_joint_relations (use --dataset anytop_truebones)"
   694	                )
   695	            for layer in self.graph_temporal_layers:
   696	                feats = layer(
   697	                    feats,
   698	                    batch.anytop_graph_dist,
   699	                    batch.anytop_joint_relations,
   700	                    batch.joint_mask,
   701	                    frame_mask_recovered,
   702	                )
   703	
   704	        fm_b = frame_mask_recovered[:, :, None, None].to(feats.dtype)
   705	        jm_b = batch.joint_mask[:, None, :, None].to(feats.dtype)
   706	
   707	        if self.feat_mode == "anytop13":
   708	            # Direct 13ch regression, AnyTop OutputProcess style: root joint
   709	            # and non-root joints use separate output Linears.
   710	            out_root = self.anytop13_head["out_root"](feats[:, :, 0:1, :])     # [B,T,1,13]
   711	            out_nonroot = self.anytop13_head["out_nonroot"](feats[:, :, 1:, :])  # [B,T,J-1,13]
   712	            pred_motion = torch.cat([out_root, out_nonroot], dim=2)            # [B,T,J,13]
   713	            pred_motion = pred_motion * fm_b * jm_b
   714	            return {
   715	                "pred_motion": pred_motion,   # [B, T, J, 13]
   716	                "pred_pos": None,
   717	                "pred_vel": None,
   718	                "frame_mask_recovered": frame_mask_recovered,
   719	            }
   720	
   721	        # fk6: TreeIK head — rest_proj + blocks + rot_head + root + FK.
   722	        rest_embed = self.treeik_head["rest_proj"](batch.rest_offsets)  # [B, J, D]
   723	        x = feats
   724	        for blk in self.treeik_head["blocks"]:
   725	            x = blk(x, rest_embed, batch.adjacency, batch.geodesic_dist,
   746	            "pred_motion": None,
   747	            "frame_mask_recovered": frame_mask_recovered,
   748	        }
   749	
   750	    def forward(self, batch: "GraphMotionBatch", sample: bool | None = None) -> dict:
   751	        """Full VAE forward: encode → decode. Returns combined dict.
   752	
   753	        Args:
   754	            sample: controls reparametrization. If None (default), uses
   755	                    self.training (sample in train, deterministic in eval).
   756	        """
   757	        if sample is None:
   758	            sample = self.training
   759	        enc = self.encode(batch, sample=sample)
   760	        dec = self.decode(enc, batch)

codex
The inference path matches the older `animate_anytop13.py` denormalization structure. I need two more checks before judging it: whether `load_anytop13_vae` calls `eval()`, and what dtype the actual checkpoint tensors have.
exec
/usr/bin/bash -lc "nl -ba scripts/animate_anytop13.py | sed -n '80,120p'" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
    80	    ).to(dev)
    81	    vae.load_state_dict(ck["model_state_dict"], strict=True)
    82	    vae.encoder.use_name_embed = bool(ta.get("use_name_embed", False))
    83	    vae.eval()
    84	    return vae, ta
    85	
    86	
    87	def main():
    88	    ap = argparse.ArgumentParser()
    89	    ap.add_argument("--ckpt", required=True)
    90	    ap.add_argument("--out", required=True)
    91	    ap.add_argument("--split", default="val")
    92	    ap.add_argument("--species", default="Alligator,Spider,Trex,Dragon",
    93	                    help="comma-separated object types to render")
    94	    ap.add_argument("--n_per", type=int, default=2)
    95	    ap.add_argument("--stride", type=int, default=2, help="frame subsample for gif")
    96	    ap.add_argument("--fps", type=int, default=8)
    97	    ap.add_argument("--caption_emb_cache", type=str, default=None,
    98	                    help="T5 caption cache (only matters for use_text ckpts); "
    99	                         "defaults to the ckpt's training cache if omitted")
   100	    ap.add_argument("--anytop_root", type=str, default=None,
   101	                    help="AnyTop processed-data root; defaults to the ckpt's "
   102	                         "training root if omitted")
   103	    ap.add_argument("--device", default="cuda")
   104	    ap.add_argument("--render_mode", choices=("rot6d", "pos", "three_col"), default="rot6d",
   105	                    help="rot6d (default): recover world via rotation channels "
   106	                         "(3:9) + bone offsets + parent chain (official FK, bone "
   107	                         "lengths exact). pos: legacy RIC path via position "
   108	                         "channels (0:3). BOTH GT and pred use the same mode so "
   109	                         "the comparison is apples-to-apples.")
   110	    args = ap.parse_args()
   111	
   112	    if args.device == "cuda" and not torch.cuda.is_available():
   113	        print("  [INFO] CUDA unavailable, falling back to CPU")
   114	        args.device = "cpu"
   115	    dev = torch.device(args.device)
   116	    out_dir = Path(args.out)
   117	    out_dir.mkdir(parents=True, exist_ok=True)
   118	
   119	    vae, ta = load_anytop13_vae(args.ckpt, dev)
   120	    print(f"Loaded anytop13 VAE: pool_type={ta['pool_type']} "

exec
/usr/bin/bash -lc "python - <<'PY'
from pathlib import Path
p=Path('scripts/_render_bf16_vae_recon_large.py')
text=p.read_text()
patterns=['rqj[:, 0] =','qm(qn(rq), rqj[:, 0])','qmv(qn(rq)','rot(pred','rot(gt']
for pat in patterns:
    print('PATTERN', pat)
    for i,line in enumerate(text.splitlines(),1):
        if pat in line:
            print(f'{i}: {line}')
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
 succeeded in 0ms:
PATTERN rqj[:, 0] =
112:     # double-root-rotation FIX (2026-06-03): REMOVED `rqj[:, 0] = qm(qn(rq), rqj[:, 0])`,
PATTERN qm(qn(rq), rqj[:, 0])
112:     # double-root-rotation FIX (2026-06-03): REMOVED `rqj[:, 0] = qm(qn(rq), rqj[:, 0])`,
PATTERN qmv(qn(rq)
95:     rp = qmv(qn(rq), rp); rp = np.cumsum(rp, axis=-2); rp[..., 1] = data[..., 1]
PATTERN rot(pred
14:   PRED_FK = rot(pred) recon rot6d channel 3:9 + FK   (orange)
283:             pred_fk = rot(pred, parents, offsets)     # PRED_FK  (recon rot6d-FK 3:9, orange)
PATTERN rot(gt
284:             gt_rot = rot(gt, parents, offsets)

exec
/usr/bin/bash -lc "python - <<'PY'
from pathlib import Path
import torch
ckpt=Path('runs/m1_bf16_anytop13_rot6dfk_w025f100t010_C128_8card_xnode_seed42/best_model.pt')
print('exists', ckpt.exists(), ckpt)
if ckpt.exists():
    ck=torch.load(ckpt, map_location='cpu', weights_only=True)
    print('keys', sorted(ck.keys()))
    sd=ck.get('model', ck.get('model_state_dict', ck.get('state_dict')))
    print('state_dict_type', type(sd).__name__, 'num_tensors', len(sd) if isinstance(sd, dict) else None)
    if isinstance(sd, dict):
        from collections import Counter
        c=Counter(str(v.dtype) for v in sd.values() if torch.is_tensor(v))
        print('dtype_counts', dict(c))
        for k,v in list(sd.items())[:8]:
            if torch.is_tensor(v): print('sample', k, v.dtype, tuple(v.shape)); break
    args=ck.get('args', {})
    print('args_subset', {k: args.get(k) for k in ['feat_mode','d_model','decoder_mode','pool_type','temporal_stride','use_text','attn_mode']})
PY" in /iridisfs/scratch/ts1v23/workspace/noKslot_bf16vae
