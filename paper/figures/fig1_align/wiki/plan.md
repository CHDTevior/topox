# plan.md — layout (canvas 1800 × 1000, y down)
Region heads y 70 (44 px bold): ① x 40 "A rig is an input" · ② x 520 "Skeleton as context" · ③ x 1060 "Shared transformer ×14" · ④ x 1420 "Predict → integrate → decode".
Region 1 (x 40–480): rest_tags crop (40,105) w 440; gharial crop (40,455) w 200; lemur crop (260,455) w 200; line "311 rigs · 34–102 joints" y 630.
Region 2 (x 520–1000): rest column x 540–600 teal, five target columns x 620–980 (w 64, gap 8), 8 rows y 300–564 (33 px pitch);
 row 5 outlined teal across all columns (the joint that receives its description vector); rest.pdf crop above the rest column;
 four noisy crops above the target columns; a teal wedge from the rest crop into the rest column; labels above (y 130) and below (y 600).
Region 3 (x 1060–1380): block panel y 110–560 with one offset copy (+10,+10) behind; sub-panels temporal y 140–255, spatial y 275–420, MLP y 440–500;
 icons at the right of each sub-panel (grid with a row / a column highlighted; the real bias matrix beside the spatial icon); clay AdaLN rail x 1075, ports at each sub-panel's left edge.
Region 4 (x 1420–1780): four clean crops y 115–195; label x̂₁ y 225–260; motion strip y 330–470; label y 495; skinned crop (1420,530) w 170; FK label x 1610 y 590–640; unseen-rig panel y 720–830.
Band (y 680–980, x 40–1000): joint-descriptions panel (teal fill) x 40–420 y 690–760; caption panel (clay fill) x 40–420 y 780–860; LLM2Vec (double outline) x 460–620 y 700–860;
 linear x 660–800 y 690–745; structure MLP x 660–800 y 762–817; caption MLP x 660–800 y 834–889; time dial (520,935); time MLP x 660–800 y 906–961; ⊕_j at (860,717); ⊕_c at (860,920).
Lines: main flow grid→block (980,430)→(1060,430) 5 px ink; block→x̂₁ (1380,160)→(1420,160); Euler loop ink 3 px (1760,108)→(1760,96)→(1020,96)→(1020,330)→(985,330) head;
 teal: rigs→rest column; desc→LLM2Vec→linear→⊕_j; struct MLP→⊕_j; ⊕_j→(900,700)→(1010,700)→(1010,465)→(985,465) head into row 5; clay: caption→LLM2Vec→MLP→⊕_c; t→MLP→⊕_c; ⊕_c→(1040,920)→(1040,520)→rail.
Occlusion: block copy under block; panels under lines? No — lines attach at panel edges and never cross panels; text last.
Word budget ~80. Everything decided at build time; draw only replays.
