# decisions.md
- D-01 (2026-09-06) Identity = the paper's own (Times, teal/clay/greys, white); Figure 1 carries no title (the caption does).
- D-02 Type floor 32 px on the 1800 px canvas (7 pt at 5.5 in); sizes 44/34/32; word budget ≈ 80; canvas 1800 × 1100 (content needed the height).
- D-03 Round 1: conditioning lanes moved under their destinations (joint language under the grid, caption + time under the
  transformer); the frozen text encoder is drawn once per lane, labelled identically — one shared box suggested joint encoding.
- D-04 Round 1: the 8-d structure MLP receives an explicit branch from the rigs (the tree); "tree structure (8-d)" labels it.
- D-05 Round 1: sub-panels sized around their icons (temporal 210, spatial 246); "tree-distance bias" sits under the mini grid.
- D-06 Round 1: the rest glyph → first column is an arrow, not a wedge; illustrations enlarged (poses 88 px, skinned crop 250 px).
- D-07 Round 2: the AdaLN rail leaves the caption+time sum directly (the sum sits under the rail); time reaches AdaLN visibly.
- D-08 Round 2: explicit output chain — prediction → (after 20 steps) motion of this rig → rotations → FK → skinned mesh — with arrows.
- D-09 Round 2: the grid arrow enters temporal attention (the first operation); the Euler return descends into a grey column from above.
- D-10 Round 2: transformer stack compressed (icons 3 rows), headings 40 px, canvas back to 1800 × 1000; skinned crop tightened.
- D-11 The blind budget (2 rounds) is spent; v4 carries the round-2 repairs and gets one closing blind reading for the record.
- D-12 (user ruling, 2026-09-06) The composition must be balanced: the joint-language lane moved UNDER the rigs (its source) and
  enters the joint row from the left; the grid grew to 10 rows and is the centrepiece; the transformer keeps its compressed height.
- D-13 (user ruling, 2026-09-06) Redesign: text too dense, colours monotonous, thin skeletons, bent lanes, box/text sizing. v6: one very
  light tint per stage band; three accents by role (teal = skeleton geometry and language, indigo = noise-mixed frames and the sampling
  loop, clay = caption and time); bold skeleton glyphs (thick round bones, small joint dots, ground shadow) from real data; lanes with at
  most one corner; boxes sized to their text at build time (measure()); lock / tree / dial icons replace "frozen" / "tree structure" /
  "flow time t"; about 70 words. The v5 arrangement (joint lane under the rigs) is kept.
- D-14 (user ruling, 2026-09-06) "Left two bands too thin, right two too dense, the big grid in band 2 is pointless and the architecture
  is not visible." v7: band 1 = the rig glyphs + "one joint, one token: 17 channels" colour strip with a legend + a small sequence strip
  (1 rest column + 240 noisy target frames) + "standardised per rig, joint, channel"; band 2 = the six addends of ONE token (linear 17→D,
  frame position, joint slot, target-frame mask, joint description→LLM2Vec→linear, tree→MLP 8-d) entering one D-dim token, with the
  caption→LLM2Vec→MLP ⊕ t→MLP → c lane under it; band 3 = ×14 transformer with temporal / spatial (tree-bias matrix) / MLP and the
  "LN · linear D→17" head; band 4 = ×20 Euler return into the token embedding, clean prediction, rotations→FK→skinned, unseen-rig pill.
  The token grid as centrepiece (D-12) is retired; the grid survives only as the small sequence strip in band 1.
- D-15 Codex design consultation (2026-09-06, read after v7): agreed on four stages and meaning-assigned accents; its one real
  warning kept — never grade colour across frame columns (it would read as noise varying over motion time). Its proposals to drop the
  numbered stage badges and to add violet→blue state colouring along the solver loop are noted, not adopted (order is real
  information; a fourth accent would break the three-role palette the user accepted).
- D-16 (user ruling, 2026-09-06) "Bands 2 and 3 are meaningless; work out the ASCII version first; no bare 'MLP' — say what it encodes
  and how it enters the model; t must sit inside the dial." wiki/ascii_flow.md is the verified flow (dit_motion.py forward, Block,
  incontext_pairs.py struct features). v8: band 2 draws the token as ⊕ of six terms, each box = input → module (17 channels → linear;
  frame index → table; joint slot → table; target frame → mask m; description → LLM2Vec 🔒 → linear; 8 tree features → MLP, features
  listed); caption → LLM2Vec 🔒 → MLP and time t → MLP sum to c. Band 3: ×14 encloses only the three sub-layers; head outside, named
  x̂₁ = predicted clean motion; an AdaLN anatomy inset (c → linear → shift, scale, gate; h → LN → ⊗ → ⊕ → f → ⊗ → ⊕ with the
  residual) under it. Band 4: Euler update node x ← x + (x̂₁ − x)/(20 − i), target frames only; the return carries the updated x.
  Canvas 1800 × 1100 (prints 3.4 in tall); the caption was shortened so the main text still ends on page 9. x̂₁ is drawn by hand
  (the render font lacks the combining hat and subscript digits).
