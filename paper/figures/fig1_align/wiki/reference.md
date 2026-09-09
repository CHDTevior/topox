# reference.md — Figure 1 of the TopX paper (ICLR 2027), ALIGN method-figure-loop trial, 2026-09-06

CRAFT_CLAIM: a Figure 1 drawn by the paper's authors in the paper's own idiom — Times labels, teal (#1F6F8B) for everything
derived from the skeleton, clay (#B5552F) for the caption/time conditioning path, grey data tokens, ink lines, white page —
in the arrangement of our own TikZ draft: four regions read left to right (a rig is an input → skeleton as context → shared
transformer ×14 → predict, integrate, decode) over a conditioning band, the rest pose visibly encoded into the first column of
one token grid, one icon per attention operation, an Euler loop over the top returning to the target frames only.

Identity: the project's (paper), not the venue's, not a reference's. Destination: a paper's Figure 1 → no title inside the
figure (the caption carries it). Composition sources (arrangement only): composition_source_tikz_v15.png (our TikZ draft, read at
5.5 in) and composition_source_codex_draft.png (a block-diagram draft by the other designer). Both are ours; there is no third-party
figure to cite.

Canvas 1800 × 1000 px; display width 5.5 in (the ICLR text width) → 327 px per inch on the canvas → type floor 7 pt = 32 px.
Type scale: 44 px bold region heads (9.6 pt), 34 px labels (7.5 pt), 32 px secondary (7 pt); nothing smaller. Face: Times
(Liberation Serif / Nimbus Roman in the headless render; the SVG names "Times New Roman, Times, serif").
Word budget: about 80 words.

Order of work (stages): 1 regions and panels · 2 the token grid · 3 illustrations (real glyph crops) · 4 connections · 5 labels.

Panel grammar: region head (bold, numbered circle); learned module = white panel, ink outline 3 px, radius 10; frozen encoder =
double ink outline; data tensor = grid of grey cells (teal cells for the rest frame); AdaLN port = small clay square on a clay
rail; sum = ink circle with +. Line grammar: main flow ink 5 px, filled head, unlabelled; skeleton path teal 3.5 px; caption/time
path clay 3.5 px; Euler loop ink 3 px over the top, labelled at mid-path with its count and guidance. Lines attach at edges.
Palette by role: teal = skeleton-derived (rest frame, joint language, tree bias, structure); clay = caption + time (modulation only);
ink = the tensor flow; grey = data cells and secondary text; white ground. No gradient, no shadow except one flat offset copy of the
block panel to say "×14".
Eye landings: 1 the token grid with its teal rest column; 2 the transformer block; 3 the fan of three rigs into the grid.
Template tells to avoid: equal panels, one arrow weight, decorative colour, a legend, a title, icons from a set.
Illustrations: real glyphs from the model's own output on a moose run-to-walk clip (rest pose with description tags, two other
rigs, noise-mixed frames, predicted clean frames, motion strip, the rig's real tree-distance bias matrix, a skinned render).
