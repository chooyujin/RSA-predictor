from __future__ import annotations
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import torch


# 0은 PAD/UNK로 사용 (vocab_size=21 가정)
aa_to_num = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "G": 6, "H": 7, "I": 8, "K": 9, "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
}


def build_token_list(seq: str) -> List[int]:
    seq = seq.strip().upper()
    return [aa_to_num.get(aa, 0) for aa in seq]


def pad_batch(
    token_lists: List[List[int]],
    esm_emb_lists: List[torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:

    if len(token_lists) == 0:
        raise ValueError("Empty token_lists")

    if len(token_lists) != len(esm_emb_lists):
        raise ValueError(f"token_lists and esm_emb_lists length mismatch: "
                         f"{len(token_lists)} vs {len(esm_emb_lists)}")

    lengths = [len(x) for x in token_lists]
    max_len = max(lengths)

    if esm_emb_lists[0].dim() != 2:
        raise ValueError(f"ESM embedding must be (L, esm_dim), got {tuple(esm_emb_lists[0].shape)}")
    esm_dim = int(esm_emb_lists[0].shape[1])

    tokens = np.zeros((len(token_lists), max_len), dtype=np.int64)
    esm_feat = np.zeros((len(token_lists), max_len, esm_dim), dtype=np.float32)
    mask = np.zeros((len(token_lists), max_len), dtype=bool)

    for i, (tok, esm) in enumerate(zip(token_lists, esm_emb_lists)):
        L = len(tok)

        if esm.shape[0] != L:
            raise ValueError(
                f"Length mismatch at index {i}: seq_len={L} vs esm_len={esm.shape[0]}"
            )

        tokens[i, :L] = np.asarray(tok, dtype=np.int64)
        esm_feat[i, :L] = esm.detach().cpu().numpy().astype(np.float32, copy=False)
        mask[i, :L] = True

    return tokens, esm_feat, mask, lengths


def prepare_inputs(
    seqs: Dict[str, str],
    esm_embedder: Callable[[List[str], List[str]], List[torch.Tensor]],
    keep_order: bool = True,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray, List[int]]:
    
    if len(seqs) == 0:
        raise ValueError("Empty seqs dict")

    ids = list(seqs.keys())
    if not keep_order:
        ids = sorted(ids)

    seq_list = [seqs[_id].strip().upper() for _id in ids]

    # 1) tokens
    token_lists = [build_token_list(s) for s in seq_list]

    # 2) esm embeddings (generated outside)
    esm_emb_list = esm_embedder(ids, seq_list)

    # sanity
    if len(esm_emb_list) != len(ids):
        raise ValueError(f"esm_embedder returned {len(esm_emb_list)} embeddings, expected {len(ids)}")

    # 3) pad + mask
    tokens_np, esm_np, mask_np, lengths = pad_batch(token_lists, esm_emb_list)

    return ids, tokens_np, esm_np, mask_np, lengths


# -------- Optional helper: convert numpy batch -> torch tensors --------
def to_torch_batch(
    tokens_np: np.ndarray,
    esm_np: np.ndarray,
    mask_np: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert numpy arrays to torch tensors on device.
    """
    tokens = torch.from_numpy(tokens_np).to(device)
    esm_feat = torch.from_numpy(esm_np).to(device)
    mask = torch.from_numpy(mask_np).to(device)
    return tokens, esm_feat, mask

