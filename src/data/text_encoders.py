"""Frozen sentence encoders for joint descriptions, behind one interface.

Two are supported. `distilbert` is mean-pooled DistilBERT, which is what the first pass used
and what the earlier probes were run against. `llm2vec` is LLM2Vec — a decoder-only LLM
converted into a text encoder by enabling bidirectional attention, masked next-token
pre-training and contrastive training (Behnamghader et al., 2024) — which is substantially
stronger at semantic similarity than a mean-pooled BERT.

Why the choice is made by measurement rather than by reputation: the joint-description gate
(FACTS.md C11) is a probe from text embedding to six graph-derived structural quantities,
fitted on retained rigs and scored on held ones. It is non-circular, so it can rank encoders
directly. Two earlier cosine-argmax probes could not, which is why they were discarded.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def _mean_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


class DistilBertEncoder:
    dim = 768

    def __init__(self, path: str = "checkpoints/text_encoders/distilbert-base-uncased",
                 device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.mod = AutoModel.from_pretrained(path).eval().to(device)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: Sequence[str], bs: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            b = self.tok(list(texts[i:i + bs]), padding=True, truncation=True,
                         max_length=32, return_tensors="pt").to(self.device)
            h = self.mod(**b).last_hidden_state
            out.append(_mean_pool(h, b["attention_mask"]).float().cpu())
        return torch.cat(out).numpy().astype(np.float32)


class LLM2VecEncoder:
    """LLM2Vec = base decoder LLM + MNTP LoRA + (optionally) a supervised LoRA.

    The supervised checkpoint expects an instruction on the QUERY side only; joint descriptions
    are documents, so they are encoded with an empty instruction, which is the library's
    convention for the document side.
    """

    def __init__(self,
                 base: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 mntp: str = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
                 supervised: str | None =
                 "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
                 device: str = "cuda", dtype=torch.bfloat16, max_length: int = 64):
        # peft probes `torch.distributed.tensor.DTensor`, but that submodule is not imported
        # by `import torch`, so the attribute lookup raises before any model is touched.
        try:
            import torch.distributed.tensor  # noqa: F401
        except Exception:
            pass
        from llm2vec import LLM2Vec

        # Use the library's own loader. Going through transformers' AutoModel with
        # trust_remote_code makes it look for `modeling_llama_encoder.py` in the HF repo, which
        # is not there — the bidirectional model classes live in the llm2vec package itself, and
        # on an offline compute node that lookup is a hard failure.
        self.l2v = LLM2Vec.from_pretrained(
            mntp,
            peft_model_name_or_path=supervised,
            device_map=device,
            torch_dtype=dtype,
            pooling_mode="mean",
            max_length=max_length,
        )
        self.tok = self.l2v.tokenizer
        self.dim = int(self.l2v.model.config.hidden_size)
        self.max_length = int(max_length)

    @torch.no_grad()
    def encode(self, texts: Sequence[str], bs: int = 64) -> np.ndarray:
        # A document is passed as a bare string; a query would be ["instruction", "text"].
        e = self.l2v.encode(list(texts), batch_size=bs)
        e = e.float().cpu().numpy() if torch.is_tensor(e) else np.asarray(e, np.float32)
        return e.astype(np.float32)

    @torch.no_grad()
    def encode_tokens(self, texts: Sequence[str], bs: int = 1):
        """Per-token hidden states for the SENTENCE span, packed to the front, plus their mask.

        Returns (H, M): H [N, max_length, dim] float32, M [N, max_length] bool. Row i holds
        `M[i].sum()` real vectors at positions `[0, M[i].sum())` and zeros after, so the caller
        gets a right-padded tensor regardless of how the library padded internally.

        Everything below is a property of the llm2vec library that was established by measurement,
        not by reading the docs, because each one silently produces a plausible-looking cache of
        the wrong text:

        * The input must go through `_convert_to_str("", text)`, which is what `encode` does to a
          bare string. That inserts the library's `!@#$%^&*()` separator, which is the only thing
          that makes `texts_2` non-empty inside `tokenize` and therefore makes `embed_mask` mark
          the sentence tokens. Passing `prepare_for_tokenization(text)` directly leaves
          `embed_mask` identically zero, and the pooled vector then averages the Llama-3 chat
          header in as well: measured 2.80 max-abs deviation, norm 123.1 against `encode`'s 131.9.
          It also applies the `doc_max_length` word-level pre-truncation.
        * Pooling is over `embed_mask & attention_mask` — the sentence span — matching
          `_skip_instruction` + `get_pooling`. The chat scaffolding is deliberately excluded.
        * `bs` defaults to 1 and should stay there. llm2vec pads on the LEFT, while HF's Llama
          derives `position_ids` from `arange(seq_len)` without discounting padding, so a batched
          row sits at RoPE positions `[n_pad, n_pad+len)` and its embedding depends on the longest
          neighbour in its chunk. Measured against per-string `encode`: bs=1 gives 3.1e-2 max abs
          (1.9 %, i.e. bf16 noise), bs=8 gives 2.1e-1 (12.7 %), bs=32 gives 1.2e-1 (7.2 %).
          Raising `bs` trades correctness for throughput; shard across GPUs instead.
        """
        from llm2vec.llm2vec import batch_to_device
        dev = next(self.l2v.model.parameters()).device
        L, D = self.max_length, self.dim
        H = np.zeros((len(texts), L, D), dtype=np.float32)
        M = np.zeros((len(texts), L), dtype=bool)

        for s in range(0, len(texts), bs):
            chunk = [self.l2v.prepare_for_tokenization(self.l2v._convert_to_str("", t))
                     for t in texts[s:s + bs]]
            feat = batch_to_device(self.l2v.tokenize(chunk), dev)
            embed_mask = feat.pop("embed_mask")                   # not a model argument
            h = self.l2v.model(**feat).last_hidden_state.float().cpu()
            # The library pools over embed_mask ALONE (_skip_instruction swaps it in).
            # AND-ing with attention_mask would silently hide the one contract
            # violation tokenize() can produce — marking left padding when its
            # independently-computed k exceeds the valid length — so that case is
            # asserted dead instead of masked away (codex encoder review #2).
            emb_b = embed_mask.bool().cpu()
            att_b = feat["attention_mask"].bool().cpu()
            if bool((emb_b & ~att_b).any()):
                raise RuntimeError(
                    "encode_tokens: embed_mask marks padded positions — llm2vec's "
                    "tokenize() contract is violated for this batch; refusing to pool")
            pool_mask = emb_b
            ids = feat["input_ids"].cpu()
            for j in range(h.shape[0]):
                mask_j = pool_mask[j]
                # STRUCTURAL GATES (codex encoder review #1/#5). The mean-pool
                # consistency check downstream cannot see a shifted or reordered
                # span, so the span is verified against the token IDS here:
                # (a) the masked ids must BE the document's own tokenisation;
                # (b) the mask must be one contiguous run (llm2vec puts the
                #     document at the end of the wrapped sequence);
                # (c) nothing is silently dropped — if the document does not fit
                #     the tokenizer cap, that is the CALLER's decision to make
                #     via its truncation allowlist, not this method's to hide.
                # Expected span = the document's own tokenisation + <|eot_id|>
                # (verified: llm2vec's embed_mask marks exactly doc+EOT; the doc
                # ids inside the wrapped sequence match standalone tokenisation
                # id-for-id, no BPE boundary merge across the separator).
                doc_ids = self.tok(texts[s + j], add_special_tokens=False,
                                   return_tensors=None)["input_ids"]
                eot = self.tok.convert_tokens_to_ids("<|eot_id|>")
                expect = doc_ids + [eot]
                got_ids = ids[j][mask_j].tolist()
                if got_ids != expect[:len(got_ids)]:
                    raise RuntimeError(
                        f"encode_tokens: masked ids are not a prefix of "
                        f"document+EOT for {texts[s + j][:80]!r}; the span is "
                        f"misaligned, not merely truncated")
                if len(got_ids) < len(expect) and ids.shape[1] < self.max_length:
                    raise RuntimeError(
                        f"encode_tokens: document lost tokens "
                        f"({len(got_ids)}/{len(expect)}) without hitting the "
                        f"tokenizer cap {self.max_length} — a non-truncation "
                        f"defect for {texts[s + j][:80]!r}")
                nz = mask_j.nonzero().flatten()
                if len(nz) and (int(nz[-1]) - int(nz[0]) + 1) != len(nz):
                    raise RuntimeError(
                        f"encode_tokens: non-contiguous document mask for "
                        f"{texts[s + j][:80]!r}")
                sel = h[j][mask_j]                                # [n_valid, D], in order
                n = sel.shape[0]
                if n > L:
                    raise RuntimeError(
                        f"encode_tokens: {n} document tokens exceed the return "
                        f"buffer L={L}; impossible under one tokenizer cap — "
                        f"caller passed inconsistent max_length")
                if n:
                    H[s + j, :n] = sel.numpy()
                    M[s + j, :n] = True
        return H, M

    @torch.no_grad()
    def encoded_token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        """Sentence-span token count per string, without running the model.

        This is the quantity the cache capacity must cover, and it is NOT the raw tokenizer
        length: the separator/chat handling above changes it. Used to set `max_length` from the
        corpus rather than by hand.
        """
        out = np.zeros(len(texts), dtype=np.int64)
        for i, t in enumerate(texts):
            wrapped = self.l2v.prepare_for_tokenization(self.l2v._convert_to_str("", t))
            feat = self.l2v.tokenize([wrapped])
            out[i] = int((feat["embed_mask"].bool() & feat["attention_mask"].bool()).sum())
        return out

    @torch.no_grad()
    def verify_pooling_matches(self, texts: Sequence[str], atol: float = 6e-2) -> float:
        """Mean-pool `encode_tokens` and require it to reproduce `encode`. Returns the max error.

        This is the gate on the whole token path: it fails loudly if the separator handling, the
        embed_mask choice or the batch size is wrong, each of which was hit during development.
        `atol` is set just above the measured bf16 floor (3.1e-2 at bs=1 on values whose per-dim
        RMS is ~2.1); it is not tight enough to catch a subtle numerical regression, only a wrong
        one, which is what it is for.
        """
        H, M = self.encode_tokens(texts)
        pooled_from_tokens = np.stack([
            H[i][M[i]].mean(0) if M[i].any() else np.zeros(self.dim, np.float32)
            for i in range(len(texts))])
        # One string per call: batched `encode` averages over its left padding too (see
        # encode_tokens), so only the unpadded single-string case is a clean reference.
        ref = np.stack([self.encode([t])[0] for t in texts])
        err = float(np.abs(pooled_from_tokens - ref).max())
        if not np.isfinite(err) or err > atol:
            raise RuntimeError(
                f"encode_tokens does not reproduce encode: max abs err {err:.4e} > {atol}. "
                f"The token cache would not be the same text the pooled path sees.")
        return err


def build(name: str, device: str = "cpu", **kw):
    if name == "distilbert":
        return DistilBertEncoder(device=device)
    if name == "llm2vec":
        return LLM2VecEncoder(device=device, **kw)
    raise ValueError(f"unknown text encoder {name!r}; expected distilbert|llm2vec")
