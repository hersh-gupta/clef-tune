"""Token-budget batching and DDP sharding (no torch)."""

from __future__ import annotations

import random


def token_budget_batches(
    lengths: list[int], max_tokens: int, seed: int, epoch: int
) -> list[list[int]]:
    """Length-sorted batches whose padded cost (len x longest) fits max_tokens.

    Composition is identical every epoch; only the batch order is reshuffled.
    """
    batches: list[list[int]] = []
    batch: list[int] = []
    longest = 0
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        new_longest = max(longest, lengths[index])
        if batch and (len(batch) + 1) * new_longest > max_tokens:
            batches.append(batch)
            batch, new_longest = [], lengths[index]
        batch.append(index)
        longest = new_longest
    if batch:
        batches.append(batch)
    random.Random(seed * 1_000_003 + epoch).shuffle(batches)
    return batches


def shard_batches(batches: list[list[int]], rank: int, world_size: int) -> list[list[int]]:
    """Pad to a multiple of world_size (repeating from the start), then take this rank's slice."""
    padded = list(batches)
    if padded:
        while len(padded) % world_size:
            padded.append(batches[(len(padded) - len(batches)) % len(batches)])
    return padded[rank::world_size]
