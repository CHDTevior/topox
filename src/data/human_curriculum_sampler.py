"""DDP-aware train sampler with a human-upsampling curriculum.

Single shared source for both the Graph-CodeFlow backbone trainer and the
Graph-VQVAE trainer. Identical semantics to the (codex-PASS) inline copy in
scripts/train_graph_codeflow.py:

- epoch < start_epoch (or factor <= 1): behaves EXACTLY like a shuffled, drop_last
  DistributedSampler — one permutation SHARED across ranks (seed+epoch, no rank),
  then a per-rank disjoint strided shard (ranks see disjoint indices).
- epoch >= start_epoch: EACH RANK independently draws its own num_samples indices by
  weighted sampling WITH REPLACEMENT (the upsampled class weight = factor, others = 1),
  using a PER-RANK seed (seed + epoch*1009 + rank). Independent per-rank draws avoid the
  systematic cross-rank duplication a shared-draw+stride would cause under replacement
  (which would over-count the upsampled class across ranks and inflate the effective
  upweighting). The global per-step upsampled share stays ~factor-upweighted. __len__ is
  equal on all ranks. With num_replicas=1/rank=0 it also serves the single-process path.
"""
from __future__ import annotations
import torch


class HumanCurriculumSampler(torch.utils.data.Sampler):
    def __init__(self, n, is_upsampled, factor, start_epoch, num_replicas, rank, seed=42):
        self.n = int(n)
        self.is_upsampled = torch.as_tensor(list(is_upsampled), dtype=torch.bool)
        self.factor = float(factor)
        self.start_epoch = int(start_epoch)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = self.n // self.num_replicas          # drop_last
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def _active(self):
        return self.factor > 1.0 and self.start_epoch >= 0 and self.epoch >= self.start_epoch

    def __iter__(self):
        if not self._active():
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch)            # shared across ranks
            idx = torch.randperm(self.n, generator=g)[:self.total_size]
            idx = idx[self.rank:self.total_size:self.num_replicas]                  # per-rank disjoint shard
        else:
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch * 1009 + self.rank)  # per-rank
            w = torch.ones(self.n, dtype=torch.float)
            w[self.is_upsampled] = self.factor
            idx = torch.multinomial(w, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())
