"""Build paper-aligned labels for final-answer-only supervision.

The manuscript defines :math:`T_i` as the final decision tokens.  The helper
below keeps the rendered chat template intact, then unmasks only the last
tokenized occurrence of the answer.  Choosing the last occurrence matters for
datasets such as GEMeX whose reasoning sometimes repeats the answer.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch


def _ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(x) for x in values]


def _last_subsequence(row: Sequence[int], needle: Sequence[int], start: int) -> int:
    if not needle or len(needle) > len(row):
        return -1
    last = -1
    upper = len(row) - len(needle) + 1
    for i in range(max(0, start), upper):
        if list(row[i : i + len(needle)]) == list(needle):
            last = i
    return last


def final_answer_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    user_lengths: torch.Tensor,
    answers: Sequence[str],
    tokenizer: Any,
    pad_token_id: int | None,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Return labels with only final-answer spans active.

    If token boundaries cannot be verified, fail instead of guessing a suffix
    that may include EOS or rationale tokens.
    """

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if pad_token_id is not None:
        labels[labels == int(pad_token_id)] = -100
    spans: list[tuple[int, int]] = []
    for row_idx, answer in enumerate(answers):
        row_active = input_ids[row_idx][attention_mask[row_idx].bool()].tolist()
        user_len = int(user_lengths[row_idx].item())
        answer_ids = _ids(tokenizer, str(answer).strip())
        start = _last_subsequence(row_active, answer_ids, user_len)
        if start < 0:
            for prefix in (" ", "\n"):
                candidate_ids = _ids(tokenizer, prefix + str(answer).strip())
                candidate_start = _last_subsequence(row_active, candidate_ids, user_len)
                if candidate_start >= 0:
                    answer_ids, start = candidate_ids, candidate_start
                    break
        if start < 0:
            raise ValueError(f"Cannot verify final-answer token span for row {row_idx}; check chat template/truncation")
        end = start + len(answer_ids)
        labels[row_idx] = -100
        active_positions = input_ids[row_idx].new_tensor(
            [i for i, flag in enumerate(attention_mask[row_idx].tolist()) if flag]
        )
        selected = active_positions[start:end]
        if selected.numel() == 0:
            raise ValueError(f"No final-answer tokens found for row {row_idx}")
        labels[row_idx, selected] = input_ids[row_idx, selected]
        spans.append((int(start), int(end)))
    return labels, spans


def assistant_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    user_lens: torch.Tensor,
    pad_token_id: int | None,
) -> torch.Tensor:
    """Mask only the user/padding prefix while retaining the full assistant.

    The final PDF restricts the *selection score* to final-answer tokens, but
    its fine-tuning protocol exports the supervised answer (including an
    optional rationale).  This helper keeps that distinction explicit.
    """
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for row, prefix_len in enumerate(user_lens.tolist()):
        positions = attention_mask[row].bool().nonzero(as_tuple=True)[0]
        labels[row, positions[:int(prefix_len)]] = -100
    return labels
