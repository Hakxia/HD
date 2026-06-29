import math

import torch


def _hidden_count(num_groups: int, mask_ratio: float) -> int:
    if num_groups < 2:
        raise ValueError("num_groups must be at least 2.")
    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be in (0, 1).")
    hidden = int(round(mask_ratio * num_groups))
    if hidden < 1 or hidden >= num_groups:
        raise ValueError(
            f"mask_ratio={mask_ratio} hides {hidden} groups for num_groups={num_groups}; "
            "at least one hidden and one visible group are required."
        )
    return hidden


def sample_hidden_mask(
    batch_size: int,
    num_groups: int,
    mask_ratio: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    hidden = _hidden_count(num_groups, mask_ratio)
    scores = torch.rand(batch_size, num_groups, device=device, generator=generator)
    indices = scores.topk(hidden, dim=1).indices
    mask = torch.zeros(batch_size, num_groups, device=device, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask


def build_balanced_mask_bank(
    num_masks: int,
    num_groups: int,
    mask_ratio: float,
    seed: int,
) -> torch.Tensor:
    if num_masks <= 0:
        raise ValueError("num_masks must be positive.")
    hidden = _hidden_count(num_groups, mask_ratio)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(num_groups, generator=generator)

    total_hidden = num_masks * hidden
    repeats = int(math.ceil(total_hidden / num_groups))
    schedule = permutation.repeat(repeats)[:total_hidden]

    bank = torch.zeros(num_masks, num_groups, dtype=torch.bool)
    for row in range(num_masks):
        cols = schedule[row * hidden : (row + 1) * hidden]
        if torch.unique(cols).numel() != hidden:
            raise RuntimeError("Internal mask bank construction produced duplicate groups in one mask.")
        bank[row, cols] = True

    row_counts = bank.sum(dim=1)
    if not torch.all(row_counts == hidden):
        raise RuntimeError("Each mask bank row must hide the same number of groups.")

    col_counts = bank.sum(dim=0)
    if int(col_counts.max() - col_counts.min()) > 1:
        raise RuntimeError("Mask bank column coverage is not balanced.")

    if num_masks == 8 and num_groups == 16 and abs(mask_ratio - 0.5) < 1e-12:
        if not torch.all(col_counts == 4):
            raise RuntimeError("Default mask bank must hide every group exactly 4 times.")

    return bank
