# KTJD review summary

The initial `[T,J_phys+1,15]` WORLD-node route was rejected after the user fixed the
meaning of TJD: `J` is the maximum number of physical joints. The accepted route is
`KTJD-17`, a physical-only `[T,J_max,17]` tensor.

The clean-context reviewer required four refinement rounds. The final design received
**9.6/10 PASS**, with no blocking or major findings. The main changes were:

- common 13 channels plus four explicitly masked physical-root channels;
- raw-source global rest-delta rotations with a literal column-cont6d codec;
- uniform canonical joint-proxy ground contact across source families;
- audited per-rig heading payload and dynamic heading validity;
- animated versus source-fixed rotation DOF separation;
- unique mask derivation, literal artifact keys, and fixed/two-pass QA gates;
- exact rest-local, yaw, FPS, origin, scale, and source-basis contracts.

