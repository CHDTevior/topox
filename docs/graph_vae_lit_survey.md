# Graph-SALAD Literature Survey

**Date:** 2026-05-20
**Scope:** Prior art for Graph-SALAD (dynamic graph pooling + multi-topology motion VAE/diffusion, J=18-143). Companion to `/scratch/ts1v23/workspace/motion_representation_study/literature_survey.md` (motion-token latent + multi-skeleton motion). This file fills the graph-side: pooling, structural-bias attention, graph diffusion.

---

## 1. Dynamic / Hierarchical Graph Pooling

| Paper | Year | Venue | Takeaway | Why-relevant |
|---|---|---|---|---|
| **DiffPool** (1806.08804) | 2018 | NeurIPS | Soft cluster matrix S: N→K, pooled A' = S^T A S. | Cleanest precedent for **soft local assignment + pooled adjacency** in DynamicGraphPool. |
| **Graph U-Nets / gPool-gUnpool** (1905.05178) | 2019 | ICML | Score node by projection on learned vector p; keep top-k. gUnpool scatters at saved indices. | Direct ancestor of our **anchor-select + unpool index restore** (mirrors DynamicGraphUnpool). |
| **SAGPool** (Lee et al.) | 2019 | ICML | Top-k pool with **GNN-computed attention scores** fusing features + topology. | Topology-aware anchor scorer for our pool. |
| **MinCutPool** (1907.00481) | 2020 | ICML | Soft assignment trained on relaxed normalised min-cut + orthogonality reg → balanced clusters, clean coarse A. | **Auxiliary loss** so clusters respect limb/tail sub-trees. |
| **Graph Multiset Transformer** (2102.11533) | 2021 | ICLR | Multi-head attention global pool; injective + permutation-invariant; strong on graph recon/gen. | **Set-style pool head** that survives variable J=18→143. |
| **Graph Parsing Networks** (2402.14393) | 2024 | ICLR | Bottom-up edge contractions; coarsening depth data-driven. | **Adaptive-depth** counterpoint to fixed 2-stage pool. |
| **Skeletal Pool/Unpool** (Aberman, 2005.05732) | 2020 | SIGGRAPH | Edge-merging pool/unpool on **homeomorphic** skeletons; tied to primal skeleton. | **Closest motion-domain ancestor**; homeomorphism limit is what we lift. |

## 2. Multi-topology / Topology-agnostic Motion

| Paper | Year | Venue | Takeaway | Why-relevant |
|---|---|---|---|---|
| **Skeleton-Aware Networks** (Aberman, 2005.05732) | 2020 | SIGGRAPH | Cross-character retarget via skeletal conv/pool to primal skeleton. | Encoder prior; homeomorphism limit. |
| **SAME** (Lee et al.) | 2023 | SIGGRAPH | Skeleton-agnostic motion embedding; topology-conditioned enc/dec. | Embedding alignment for mixed-J batches. |
| **AnyTop** (2502.17327) | 2025 | SIGGRAPH | Diffusion on arbitrary non-homeo skeletons; skeleton feats = rest-pose + relations + geodesic + joint text; topology-biased attn. | **Direct sibling of our denoiser.** Read for bias formulation. |
| **UniMoGen** (2505.21837) | 2025 | – | UNet diffusion, joints as axis, topology mask in attn, no padding. | Counter-design ablation target. |
| **Topology-Agnostic Animal** (2512.10352) | 2025 | – | 140 species, RVQ-VAE + graph-transformer with topology bias. | Upper-J multi-species precedent. |

## 3. Graph-conditioned Diffusion

| Paper | Year | Venue | Takeaway | Why-relevant |
|---|---|---|---|---|
| **DiGress** (2209.14734) | 2023 | ICLR | Discrete denoise over categorical node/edge; marginal-preserving Markov; graph-transformer denoiser; scales to 1.3M-mol GuacaMol. | Canonical **graph-structure-as-data** diffusion; denoiser blueprint. |
| **GraphGDP** (Huang et al.) | 2022 | ICDM | Position-aware score-based continuous graph diffusion. | Continuous alternative; closer to SALAD's latent setting. |
| **EDM / GeoLDM** (Hoogeboom; Xu) | 2022-23 | ICML/ICLR | Equivariant diffusion on 3D molecules; latent variants. | **Diffusion in latent conditioned on 3D graph** — our exact recipe. |
| **MDM** (2209.14916) | 2023 | ICLR | Transformer denoiser on raw motion + CLIP text. | Ancestor baseline. |
| **MLD** (2212.04048) | 2023 | CVPR | **Latent** motion diffusion (denoiser over VAE latent). | Our Phase 1+2 stack follows MLD; we add graph conditioning. |
| **SALAD** (2503.13836) | 2025 | CVPR | Skeleton-aware VAE w/ decoupled spatial/temporal + diffusion denoiser; zero-shot editing via attn. | **Paper we extend.** Fixed-22-joint is what we lift. |

## 4. Graph Attention with Structural Bias

| Paper | Year | Venue | Takeaway | Why-relevant |
|---|---|---|---|---|
| **Graphormer** (2106.05234) | 2021 | NeurIPS | `attn_ij += b(SPD(i,j)) + edge_enc + centrality`; SOTA OGB-LSC. | **Reference formulation** for §10.3 `attn += adj_bias + geo_bias`. Most-cited. |
| **GRPE** (2201.12787) | 2022 | ICLR-MLDD | Q/K/V each see node-topology + node-edge relative encodings; no linearisation. | Cleaner than Graphormer for **bone-edge features into Q/K/V**. |
| **GraphGPS** (2205.12454) | 2022 | NeurIPS | Modular GT: local MPNN + global attn + positional/structural enc. | Template if pool needs local+global mix per layer. |
| **SAN** (Kreuzer et al.) | 2021 | NeurIPS | Laplacian-eigvec positional enc; attn over real/virtual edges. | **Alt to SPD** when geodesic is degenerate (snake/dragon long chains). |
| **EGT** (Hussain et al.) | 2022 | KDD | Edges as first-class tokens w/ own pair-attn channel. | Bone-type edges carry direction + length semantics. |

## 5. Motion VAE / Diffusion Baselines (Context)

- **SALAD** (2503.13836, CVPR 2025) — parent. **MLD** (2212.04048, CVPR 2023) — latent-diffusion ancestor; Phase 1+2 stack.
- **MDM** (2209.14916, ICLR 2023) — raw-motion diffusion baseline. **MotionDiffuse** (2208.15001, 2022) — earliest large text-to-motion diffusion.
- **TEMOS** (2204.14109, ECCV 2022) — T2M VAE, KL-warmup recipe. **ACTOR** (2104.05670, ICCV 2021) — action-cond VAE, transformer-VAE template.
- **T2M-GPT** (2301.06052, CVPR 2023) — VQ+GPT discrete-token baseline. **MotionGPT** (2306.14795, NeurIPS 2023) — LLM-style motion.
- **HumanML3D** (Guo et al., CVPR 2022) — dataset + eval conventions (FID, MM-Dist, R-precision).

---

## Design Recommendations

1. **DynamicGraphPool = DiffPool soft-assignment + SAGPool topology-aware anchor scoring + MinCutPool aux loss.** DiffPool gives pooled-adjacency machinery; SAGPool gives topology-aware anchor score; MinCut + orthogonality stops cluster collapse across limb boundaries at extreme J (Dragon 143, Anaconda 27). Aberman's skeletal pool is the right *intuition* but too rigid (homeomorphic).
2. **Phase 2 graph-aware skeletal attention = Graphormer SPD bias + GRPE-style edge-aware K/V**, not vanilla learned bias. Geodesic SPD is the right structural prior; GRPE handles bone-type edge features.
3. **DynamicGraphUnpool = gUnpool index restore + light feature refine**, mirroring Graph U-Nets. Save anchor indices + assignment matrix at pool; unpool = scatter + learned refine. Symmetric stack is what makes recon tractable at J=143.
4. **Use MLD's latent-diffusion recipe as Phase 2 backbone** (denoiser over VAE latent). Immediate SALAD ancestor; clean comparison. Add DiGress-style discrete graph-structure conditioning only if Phase 3 unseen-topology gen demands it.

---

**Full file:** `/scratch/ts1v23/workspace/noKslot_clean/docs/graph_vae_lit_survey.md`
**Companion (motion-side, prior):** `/scratch/ts1v23/workspace/motion_representation_study/literature_survey.md`
