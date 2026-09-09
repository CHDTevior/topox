## Round 1 (v1) — blind, gpt-6-astra, 2026-09-06 — score 6/10
Reading recovered: rig input, rest column + noise-mixed targets, description added along a joint row, caption + t → AdaLN
(modulation only), temporal/spatial/MLP with tree bias, prediction → Euler loop → target frames, decode chain, unseen-rig note.
Kept observations (verbatim):
- "The structure MLP has no incoming connection, so the figure does not identify which skeleton information it consumes." → primitive: draw the tree → structure MLP branch.
- "recompose the conditioning paths directly beneath their destinations—joint conditioning beneath the grid, caption/time beneath the transformer" → primitive: two lanes, two frozen-encoder boxes, AdaLN rail on the block's right.
- "The temporal-attention minigrid crosses its panel's bottom border. 'Tree bias' extends past the spatial panel's right border." → parameter: panel heights, label under the mini grid.
- "The noisy poses are only roughly 30 canvas px tall … The pale skinned animal also occupies only a small part" → parameter: 88-px poses, 250-px skinned crop.
- "The rest column has an unexplained decorative-looking cap." → parameter: wedge replaced by an arrow.
- "The black input arrow meets the block beside spatial attention … without internal arrows, the execution order is not explicit." → parameter: internal arrows temporal → spatial → MLP; output leaves at the MLP.
- "s = 2 is not defined … whether LoRA is optional or required" → wording: "(s = 2)" kept for the caption; "optional LoRA, ~10 clips".
v2 redrawn with all of the above; render awaits blind round 2.
- v2 rendered; blind round 2 launched (parts 1 and 2)
## Round 2 (v2) — blind, gpt-6-astra — score 6/10 (part 1 read the central method correctly)
Kept observations: "The AdaLN spine appears to join the caption branch above the bottom plus … time never reaches AdaLN" (fixed v3:
the rail leaves the sum); "connect the returned target state to the loop and connect the final result to the decoding sequence"
(v3/v4: explicit arrows prediction → after 20 steps → motion → FK → skinned; loop enters the grey columns from above); "the transformer
entrance … nothing visibly carries the input to temporal attention" (v4: the input arrow enters the temporal box); "shorten the
transformer stack" (v4: −25%); "put the frame-count bracket under the grey columns and label the rest column separately" (v4);
"the spatial heatmap touches the panel's right boundary" (v4: inset). v4 rendered; a final blind reading (round 3, part 1) launched.
## Round 3 (v4) — blind reading only, gpt-6-astra — score 8/10
Reading recovered every promised relationship; remaining: "the return line does not explicitly show the operation between the clean
prediction and the updated target frames, and s = 2 is undefined" (caption states s); "make the output match the motion claim visibly"
(deferred: the strobe figure carries it); "bring joint callouts, 'frozen', 'AdaLN', '×14' to at least 32 px" (kept at 26 for space; the
paper reads them at 5.7 pt). v5 rebalanced the composition after the user's ruling (D-12); not blind-reviewed.
- v6 redesign per user ruling (D-13); not blind-reviewed; codex design consultation running
## Round 4 (v6) — blind reading, gpt-6-astra — score 7/10
Read the conditioning paths and the two attention directions correctly. Misses that v7 answers: the head's output and what returns in
the loop are unnamed (v7: "LN · linear D→17" head; the Euler return enters "linear 17→D"); attention labels overrun their boxes and
"tree bias"/"AdaLN" are too small (v7: transformer panel widened to 360 px, labels 30–34 px). Still open: no explicit Euler-update box
(the caption states the update and s = 2); LoRA is a pill, not a drawn branch.
- v7 (D-14) rendered clean at 5.5 in (300 dpi in-paper crop review/inpaper_fig1_300dpi.png); exported SVG→PDF and wired into the paper;
  blind round 5 launched on whole_v7.png (asks explicitly about architecture visibility, type size, mis-routable lines, density balance).
## Round 5 (v7) — blind reading, gpt-6-astra — score 7/10
Converged with the user's critique: token assembly ambiguous (fusion operator not drawn), "MLP (8-d)" and padlocks unclear, ×14 boundary
wrongly includes the head, head's target unnamed, no Euler-update node, small secondary labels, band 2 denser than its neighbours and
band 3's bottom quarter empty. All addressed in v8 (D-16); round 6 launched on whole_v8.png.
## Round 6 (v8) — blind reading, gpt-6-astra — score 8/10 ("yes, the reader now sees a concrete model")
Read back correctly: the six-term sum, the two attention axes, the tree bias, the clean-motion head, the update node and that the
UPDATED x (not x̂₁) returns. Fixed in v8d: the tree-MLP output arrow had zero length (box touched the bus; now the box is narrower and
shifted left); the inset footer wrongly implied the head has a gate and residual (now "the head takes shift and scale only");
feed-forward panel got "per token, D→4D→D"; the update node states i = 0…19; the dial is labelled "flow time t". Not taken: "rotation Δ
reference", "what the mask vector is" (caption territory, no room), widening bands 2–3 at the expense of the render (the user chose the
architecture content; the render is the output evidence). The review PNG on the figure page is now rendered from figure1.pdf itself.
- v8e/v8f (user, 2026-09-06 afternoon): "tree-features box overlaps its arrow; the temporal/spatial attention icons and the tree-bias
  matrix are ugly and hard to read; their text and the FFN text run out of their boxes; the update box meets its arrows badly."
  Fixes: the attention icons are now tree glyphs (temporal: the same joint linked across four frames; spatial: one joint attending
  to every joint of a frame, the others shaded by hop distance, with a 1 hop → 8 hops colour bar next to "tree bias −min(hops, 8)");
  sub-panels widened (margin 20, indent 40) and the FFN panel renamed "feed-forward / per token, D→4D→D"; the tree-features box
  narrowed and moved so both its arrows have length; the update box's second line shortened so the return line leaves it with a
  visible segment. Diagnostic: PDF and browser text widths agree within 2 % (same Nimbus Roman), so overflow was geometry, not font.
  Round 7 blind reading launched on the PDF-rendered v8.
