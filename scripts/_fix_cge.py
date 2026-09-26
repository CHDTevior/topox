"""codex NEEDS-FIX #1: the emb_sink payload must carry a row key that is UNIQUE per row
(motion_id can repeat across caption rows), plus a row count, so MultiModality can prove
row i is the same text in every repeat."""
p = "src/eval/codeflow_gen_eval.py"
s = open(p).read()

# (a) allocate the row-key accumulator ONLY when a sink is requested (zero cost for the trainer)
old = """        B = max(1, gen_batch)
        done = 0
        for bstart in range(0, len(idxs), B):"""
new = """        B = max(1, gen_batch)
        done = 0
        rkeys = [] if emb_sink is not None else None   # (dataset_index, caption) — unique per ROW
        for bstart in range(0, len(idxs), B):"""
assert s.count(old) == 1, "loop-head anchor not unique"
s = s.replace(old, new)

# (b) GT-baseline path
old = """                GE.append(gte); GTE.append(gte); TE.append(core.encode_text(caps).float().cpu())
                mids.extend(str(ds._plan[di][1]["motion_id"]) for di in bidx)"""
new = """                GE.append(gte); GTE.append(gte); TE.append(core.encode_text(caps).float().cpu())
                mids.extend(str(ds._plan[di][1]["motion_id"]) for di in bidx)
                if rkeys is not None:
                    rkeys.extend((int(di), str(c)) for di, c in zip(bidx, caps))"""
assert s.count(old) == 1, "gt-baseline anchor not unique"
s = s.replace(old, new)

# (c) generation path
old = """            TE.append(core.encode_text(caps).float().cpu())
            mids.extend(str(ds._plan[di][1]["motion_id"]) for di in bidx)
            done += len(items)"""
new = """            TE.append(core.encode_text(caps).float().cpu())
            mids.extend(str(ds._plan[di][1]["motion_id"]) for di in bidx)
            if rkeys is not None:
                rkeys.extend((int(di), str(c)) for di, c in zip(bidx, caps))
            done += len(items)"""
assert s.count(old) == 1, "gen-path anchor not unique"
s = s.replace(old, new)

# (d) hand row_keys out with the embeddings
old = """        if emb_sink is not None:
            emb_sink.append({"gen_emb": GE.clone(), "motion_ids": list(mids)})"""
new = """        if emb_sink is not None:
            emb_sink.append({"gen_emb": GE.clone(), "motion_ids": list(mids),
                             "row_keys": list(rkeys)})"""
assert s.count(old) == 1, "sink anchor not unique"
s = s.replace(old, new)

open(p, "w").write(s)
print("patched", p)
