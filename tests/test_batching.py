from clef_tune.batching import shard_batches, token_budget_batches


def test_batches_cover_every_index_once_within_the_padded_budget():
    lengths = [10, 300, 50, 50, 120, 7, 300, 45]
    batches = token_budget_batches(lengths, max_tokens=400, seed=1, epoch=0)
    assert sorted(i for batch in batches for i in batch) == list(range(8))
    for batch in batches:
        assert len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 400


def test_an_oversized_item_gets_its_own_batch():
    batches = token_budget_batches([900, 10, 12], max_tokens=400, seed=0, epoch=0)
    assert [0] in batches


def test_batch_order_is_seeded_per_epoch():
    lengths = list(range(1, 41))
    first = token_budget_batches(lengths, 60, seed=3, epoch=0)
    assert first == token_budget_batches(lengths, 60, seed=3, epoch=0)
    second = token_budget_batches(lengths, 60, seed=3, epoch=1)
    assert first != second and sorted(first) == sorted(second)


def test_shard_batches_pads_ranks_to_equal_length():
    batches = [[0], [1], [2], [3], [4]]
    assert [shard_batches(batches, r, 2) for r in range(2)] == [[[0], [2], [4]], [[1], [3], [0]]]
    assert shard_batches(batches, 0, 1) == batches
