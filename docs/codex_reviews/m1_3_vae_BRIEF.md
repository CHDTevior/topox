# M1.3 VAE round 1

`--NEEDS-FIX`

M1.3 step 2 gate: `NO-GO`.

1. F1: ckpt-compatible decoder load is blocked. Baseline `Model` stores base decoder weights as `decoder.*`, but `GraphMotionVAE` stores the MotionDecoder under `decoder.base.*`; direct load gives missing/unexpected decoder keys. Also baseline TreeIK uses default 8 heads, while VAE passes base `n_heads` into TreeIK.

2. F2: bitwise M1.0 no-pool mode is not expressible. Current `pool_type='none'` still applies temporal AvgPool, `Linear(D,2D)`, random reparameterization, and repeat upsample. Add an explicit identity/deterministic latent path for the ckpt gate.

3. F3: R12 fail-loud gaps remain. VAE validates `pool_type` and `temporal_stride`, but not dropout range, `d_model % n_heads`, or no-pool `T % temporal_stride`; invalid no-pool T currently fails with raw PyTorch RuntimeErrors.

Key findings: happy-path pipeline shapes are otherwise correct: encoder/SlotNorm/pool, assignment->unpool, and TopoFK `cat([pos, vel])` split all line up; tests pass locally (`python tests/test_vae.py`, 6 OK).
