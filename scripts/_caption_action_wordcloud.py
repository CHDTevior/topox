#!/usr/bin/env python
"""One-off: action-vocabulary word cloud from L5 motion captions.

Removes articles/prepositions (English stopwords), sex/age qualifiers, and
animal-name tokens (from object names) so the remaining frequency-weighted
words are the ACTION / behaviour vocabulary the dataset describes.
"""
import csv, json, re, collections, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

ROOT = "/scratch/ts1v23/workspace/noKslot_clean/data/animo4d_anytop_clean_L5"
OUT = "/scratch/ts1v23/workspace/noKslot_clean/analysis_caption_wordcloud/L5_action_wordcloud.png"

rows = list(csv.DictReader(open(f"{ROOT}/motion_text_manifest.csv", newline="")))

# Collect all caption strings (text + texts list).
caps = []
animal_tokens = set()
for r in rows:
    on = (r.get("object_name") or "")
    for tok in re.split(r"[_\s]+", on.lower()):
        if tok and tok != "pz":
            animal_tokens.add(tok)
    t = (r.get("text") or "").strip()
    if t:
        caps.append(t)
    tt = (r.get("texts") or "").strip()
    if tt:
        try:
            for x in json.loads(tt):
                if isinstance(x, str) and x.strip():
                    caps.append(x.strip())
        except Exception:
            for x in re.split(r"[;|]", tt):
                if x.strip():
                    caps.append(x.strip())

# Stopwords: English defaults + sex/age + animal names + a few fillers.
stop = set(STOPWORDS)
stop |= {"male", "female", "juvenile", "adult", "the", "a", "an", "and", "then",
         "to", "of", "its", "it", "his", "her", "their", "on", "in", "at", "by",
         "as", "with", "for", "from", "while", "before", "after", "finally",
         "gradually", "begins", "starts", "start", "begin", "part", "gimmick"}
stop |= animal_tokens  # reticulated, giraffe, aardvark, ... (not actions)

# Count word frequencies over the action vocabulary.
freq = collections.Counter()
wre = re.compile(r"[a-z]+")
for c in caps:
    for w in wre.findall(c.lower()):
        if len(w) <= 1 or w in stop:
            continue
        freq[w] += 1

print(f"captions={len(caps)}  unique_action_words={len(freq)}  animal_tokens_removed={len(animal_tokens)}")
print("=== top 35 action words ===")
for w, n in freq.most_common(35):
    print(f"  {w:18s} {n:7d}")

wc = WordCloud(width=1800, height=1000, background_color="white",
               colormap="viridis", max_words=160, prefer_horizontal=0.9,
               collocations=False, min_font_size=8)
wc.generate_from_frequencies(dict(freq.most_common(160)))

import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
plt.figure(figsize=(18, 10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("AniMo4D L5 motion-caption ACTION vocabulary  (word size ∝ frequency; animal names / sex / articles removed)",
          fontsize=15, pad=12)
plt.tight_layout()
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print("SAVED:", OUT)
