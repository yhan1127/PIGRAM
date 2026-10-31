# src/pigram/models/encoders.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pigram.utils.constants import AMINO_ACID_VOCAB


class ProteinPatchEncoder(nn.Module):
    """
    Protein patch encoder.

    Returns:
        {
            "z_patch": [N_patches, hidden_dim],
            "patch_attr": [N_patches, 56],
            "patch_summary_feat": [N_patches, 56],
        }

    If precomputed_patch_attr is provided, explicit 56D patch attributes are
    read directly from it. Otherwise, the same 56D attributes are computed
    on the fly for backward compatibility.
    """

    def __init__(
        self,
        protein_vocab_size: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        seq_offset_clip: int = 32,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.seq_offset_clip = int(seq_offset_clip)
        self.patch_attr_dim = 56

        self.token_emb = nn.Embedding(
            num_embeddings=protein_vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0,
        )

        self.rel_dist_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.seq_offset_scalar_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.seq_offset_bucket_emb = nn.Embedding(
            num_embeddings=2 * seq_offset_clip + 1,
            embedding_dim=hidden_dim,
        )

        self.center_flag_emb = nn.Embedding(
            num_embeddings=2,
            embedding_dim=hidden_dim,
        )

        self.node_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_value_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.node_gate_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.center_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.summary_encoder = nn.Sequential(
            nn.Linear(self.patch_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.register_buffer(
            "aa_ids_pos",
            torch.tensor(
                [
                    AMINO_ACID_VOCAB["K"],
                    AMINO_ACID_VOCAB["R"],
                    AMINO_ACID_VOCAB["H"],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_neg",
            torch.tensor(
                [
                    AMINO_ACID_VOCAB["D"],
                    AMINO_ACID_VOCAB["E"],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_aromatic",
            torch.tensor(
                [
                    AMINO_ACID_VOCAB["F"],
                    AMINO_ACID_VOCAB["Y"],
                    AMINO_ACID_VOCAB["W"],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_hydrophobic",
            torch.tensor(
                [
                    AMINO_ACID_VOCAB["A"],
                    AMINO_ACID_VOCAB["V"],
                    AMINO_ACID_VOCAB["I"],
                    AMINO_ACID_VOCAB["L"],
                    AMINO_ACID_VOCAB["M"],
                    AMINO_ACID_VOCAB["F"],
                    AMINO_ACID_VOCAB["W"],
                    AMINO_ACID_VOCAB["Y"],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_polar",
            torch.tensor(
                [
                    AMINO_ACID_VOCAB["S"],
                    AMINO_ACID_VOCAB["T"],
                    AMINO_ACID_VOCAB["N"],
                    AMINO_ACID_VOCAB["Q"],
                    AMINO_ACID_VOCAB["C"],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_gly",
            torch.tensor([AMINO_ACID_VOCAB["G"]], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_pro",
            torch.tensor([AMINO_ACID_VOCAB["P"]], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_small",
            torch.tensor(
                [
                    AMINO_ACID_VOCAB["A"],
                    AMINO_ACID_VOCAB["G"],
                    AMINO_ACID_VOCAB["S"],
                    AMINO_ACID_VOCAB["C"],
                    AMINO_ACID_VOCAB["P"],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "aa_ids_bulky",
            torch.tensor(
                [
                    AMINO_ACID_VOCAB["F"],
                    AMINO_ACID_VOCAB["W"],
                    AMINO_ACID_VOCAB["Y"],
                    AMINO_ACID_VOCAB["R"],
                    AMINO_ACID_VOCAB["K"],
                    AMINO_ACID_VOCAB["L"],
                    AMINO_ACID_VOCAB["I"],
                    AMINO_ACID_VOCAB["M"],
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _ratio_for_group(
        self,
        patch_token_ids: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        mask = (
            patch_token_ids.unsqueeze(-1)
            == group_ids.view(1, 1, -1)
        ).any(dim=-1).float()

        return mask.mean(dim=1, keepdim=True)

    def _compute_patch_aa_composition(
        self,
        patch_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        one_hot = F.one_hot(
            patch_token_ids.clamp(min=0, max=21),
            num_classes=22,
        ).float()

        return one_hot[:, :, 1:22].mean(dim=1)

    def _compute_patch_summary_features(
        self,
        patch_token_ids: torch.Tensor,
        patch_indices: torch.Tensor,
        patch_center_indices: torch.Tensor,
        patch_relative_distances: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8

        center_idx = patch_center_indices.unsqueeze(1)
        seq_offsets = patch_indices - center_idx
        abs_seq_offsets = seq_offsets.abs().float()

        dists = patch_relative_distances.float()

        is_center = patch_indices == center_idx
        non_center_mask = (~is_center).float()
        non_center_count = non_center_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        aa_comp = self._compute_patch_aa_composition(patch_token_ids)

        frac_pos = self._ratio_for_group(patch_token_ids, self.aa_ids_pos)
        frac_neg = self._ratio_for_group(patch_token_ids, self.aa_ids_neg)
        frac_aromatic = self._ratio_for_group(patch_token_ids, self.aa_ids_aromatic)
        frac_hydrophobic = self._ratio_for_group(patch_token_ids, self.aa_ids_hydrophobic)
        frac_polar = self._ratio_for_group(patch_token_ids, self.aa_ids_polar)
        frac_gly = self._ratio_for_group(patch_token_ids, self.aa_ids_gly)
        frac_pro = self._ratio_for_group(patch_token_ids, self.aa_ids_pro)
        frac_small = self._ratio_for_group(patch_token_ids, self.aa_ids_small)
        frac_bulky = self._ratio_for_group(patch_token_ids, self.aa_ids_bulky)

        chemistry_feat = torch.cat(
            [
                frac_pos,
                frac_neg,
                frac_aromatic,
                frac_hydrophobic,
                frac_polar,
                frac_gly,
                frac_pro,
                frac_small,
                frac_bulky,
            ],
            dim=-1,
        )

        dist_mean = dists.mean(dim=1, keepdim=True)
        dist_std = dists.std(dim=1, keepdim=True, unbiased=False)
        dist_min = dists.min(dim=1, keepdim=True).values
        dist_max = dists.max(dim=1, keepdim=True).values
        dist_median = dists.median(dim=1, keepdim=True).values

        non_center_dists = dists * non_center_mask
        non_center_dist_mean = non_center_dists.sum(dim=1, keepdim=True) / non_center_count

        centered_non_center = (dists - non_center_dist_mean) * non_center_mask
        non_center_dist_std = torch.sqrt(
            (
                centered_non_center.pow(2).sum(dim=1, keepdim=True)
                / non_center_count
            ).clamp_min(eps)
        )

        non_center_dist_max = non_center_dists.max(dim=1, keepdim=True).values

        non_center_dist_min = torch.where(
            non_center_mask > 0,
            dists,
            torch.full_like(dists, 1e9),
        ).min(dim=1, keepdim=True).values
        non_center_dist_min = torch.where(
            torch.isfinite(non_center_dist_min),
            non_center_dist_min,
            torch.zeros_like(non_center_dist_min),
        )

        frac_within_4a = (dists <= 4.0).float().mean(dim=1, keepdim=True)
        frac_within_8a = (dists <= 8.0).float().mean(dim=1, keepdim=True)

        distance_feat = torch.cat(
            [
                dist_mean,
                dist_std,
                dist_min,
                dist_max,
                dist_median,
                non_center_dist_mean,
                non_center_dist_std,
                non_center_dist_min,
                non_center_dist_max,
                0.5 * (frac_within_4a + frac_within_8a),
            ],
            dim=-1,
        )

        mean_abs_offset = abs_seq_offsets.mean(dim=1, keepdim=True)
        std_abs_offset = abs_seq_offsets.std(dim=1, keepdim=True, unbiased=False)
        max_abs_offset = abs_seq_offsets.max(dim=1, keepdim=True).values

        frac_local_2 = (abs_seq_offsets <= 2).float().mean(dim=1, keepdim=True)
        frac_local_4 = (abs_seq_offsets <= 4).float().mean(dim=1, keepdim=True)
        frac_local_8 = (abs_seq_offsets <= 8).float().mean(dim=1, keepdim=True)
        frac_long_16 = (abs_seq_offsets > 16).float().mean(dim=1, keepdim=True)

        signed_offset_mean = seq_offsets.float().mean(dim=1, keepdim=True)
        pos_frac = (seq_offsets > 0).float().mean(dim=1, keepdim=True)
        neg_frac = (seq_offsets < 0).float().mean(dim=1, keepdim=True)
        signed_balance = pos_frac - neg_frac

        topology_feat = torch.cat(
            [
                mean_abs_offset,
                std_abs_offset,
                max_abs_offset,
                frac_local_2,
                frac_local_4,
                frac_local_8,
                frac_long_16,
                signed_offset_mean,
                signed_balance,
            ],
            dim=-1,
        )

        net_charge = frac_pos - frac_neg
        charge_magnitude = frac_pos + frac_neg
        cation_pi_proxy = frac_pos * frac_aromatic
        aromatic_hydrophobic_packing = frac_aromatic * frac_hydrophobic
        compactness_proxy = 1.0 / (1.0 + non_center_dist_mean)
        locality_balance = frac_local_4 - frac_long_16
        asymmetry_strength = signed_balance.abs()

        derived_feat = torch.cat(
            [
                net_charge,
                charge_magnitude,
                cation_pi_proxy,
                aromatic_hydrophobic_packing,
                compactness_proxy,
                locality_balance,
                asymmetry_strength,
            ],
            dim=-1,
        )

        summary_feat = torch.cat(
            [
                aa_comp,
                chemistry_feat,
                distance_feat,
                topology_feat,
                derived_feat,
            ],
            dim=-1,
        )

        if summary_feat.shape[-1] != self.patch_attr_dim:
            raise RuntimeError(
                f"patch summary feature should be {self.patch_attr_dim}D, "
                f"got {summary_feat.shape}"
            )

        return summary_feat

    def forward(
        self,
        protein_token_ids: torch.Tensor,
        patch_indices: torch.Tensor,
        patch_center_indices: torch.Tensor,
        patch_relative_distances: torch.Tensor,
        precomputed_patch_attr: Optional[torch.Tensor] = None,
    ):
        residue_emb = self.token_emb(protein_token_ids)

        patch_token_ids = protein_token_ids[patch_indices]
        patch_res_emb = residue_emb[patch_indices]
        center_res_emb = residue_emb[patch_center_indices]

        rel_dist_emb = self.rel_dist_proj(
            patch_relative_distances.unsqueeze(-1)
        )

        seq_offsets = patch_indices - patch_center_indices.unsqueeze(1)
        seq_offsets_clipped = seq_offsets.clamp(
            min=-self.seq_offset_clip,
            max=self.seq_offset_clip,
        )
        seq_offset_bucket_ids = seq_offsets_clipped + self.seq_offset_clip
        seq_offset_bucket_emb = self.seq_offset_bucket_emb(seq_offset_bucket_ids)

        norm_seq_offsets = seq_offsets.float() / float(self.seq_offset_clip)
        norm_seq_offsets = norm_seq_offsets.clamp(-1.0, 1.0)
        seq_offset_scalar_emb = self.seq_offset_scalar_proj(
            norm_seq_offsets.unsqueeze(-1)
        )

        center_flags = (patch_indices == patch_center_indices.unsqueeze(1)).long()
        center_flag_emb = self.center_flag_emb(center_flags)

        node_feat = torch.cat(
            [
                patch_res_emb,
                rel_dist_emb,
                seq_offset_bucket_emb + seq_offset_scalar_emb,
                center_flag_emb,
            ],
            dim=-1,
        )

        node_feat = self.node_fuse(node_feat)

        node_values = self.node_value_proj(node_feat)
        node_gates = self.node_gate_proj(node_feat).squeeze(-1)
        node_weights = torch.softmax(node_gates, dim=1).unsqueeze(-1)
        neighborhood_feat = (node_values * node_weights).sum(dim=1)

        center_feat = self.center_proj(center_res_emb)

        if precomputed_patch_attr is not None:
            patch_summary_feat = precomputed_patch_attr.float()
        else:
            patch_summary_feat = self._compute_patch_summary_features(
                patch_token_ids=patch_token_ids,
                patch_indices=patch_indices,
                patch_center_indices=patch_center_indices,
                patch_relative_distances=patch_relative_distances,
            )

        if patch_summary_feat.shape[-1] != self.patch_attr_dim:
            raise RuntimeError(
                f"precomputed_patch_attr should be {self.patch_attr_dim}D, "
                f"got {patch_summary_feat.shape}"
            )

        summary_emb = self.summary_encoder(patch_summary_feat)

        z_patch = torch.cat(
            [
                center_feat,
                neighborhood_feat,
                summary_emb,
            ],
            dim=-1,
        )
        z_patch = self.out_proj(z_patch)

        return {
            "z_patch": z_patch,
            "patch_attr": patch_summary_feat,
            "patch_summary_feat": patch_summary_feat,
        }


class MotifEncoder(nn.Module):
    """
    ssNA motif encoder.

    Returns:
        {
            "z_motif": [N_motifs, hidden_dim],
            "motif_attr": [N_motifs, 27],
            "motif_summary_feat": [N_motifs, 27],
        }

    If precomputed_motif_attr is provided, explicit 27D motif attributes are
    read directly from it. Otherwise, the same 27D attributes are computed
    on the fly for backward compatibility.
    """

    def __init__(
        self,
        rna_vocab_size: int = 16,
        motif_type_vocab_size: int = 8,
        edge_side_vocab_size: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.motif_type_vocab_size = int(motif_type_vocab_size)
        self.edge_side_vocab_size = int(edge_side_vocab_size)
        self.motif_attr_dim = 27

        self.nt_embedding = nn.Embedding(
            num_embeddings=rna_vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0,
        )

        self.type_embedding = nn.Embedding(
            num_embeddings=motif_type_vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0,
        )

        self.edge_side_embedding = nn.Embedding(
            num_embeddings=edge_side_vocab_size,
            embedding_dim=hidden_dim,
            padding_idx=0,
        )

        self.length_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pairing_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gc_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.terminal_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.center_pos_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.seq_token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.seq_attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.seq_value_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.summary_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _token_ids_to_base_chars(token_ids: torch.Tensor) -> List[str]:
        base_map = {
            1: "A",
            2: "C",
            3: "G",
            4: "U",
            5: "N",
        }
        ids = token_ids.detach().cpu().long().tolist()
        return [base_map.get(int(x), "N") for x in ids if int(x) != 0]

    @staticmethod
    def _nt_fraction(base_chars: List[str], base: str) -> float:
        if len(base_chars) == 0:
            return 0.0
        return float(sum(x == base for x in base_chars) / len(base_chars))

    @staticmethod
    def _nt_entropy(frac_a: float, frac_c: float, frac_g: float, frac_u: float) -> float:
        probs = [frac_a, frac_c, frac_g, frac_u]
        entropy = 0.0
        eps = 1e-8
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p + eps)
        return float(entropy / math.log(4.0))

    @staticmethod
    def _max_run_fraction_from_chars(base_chars: List[str]) -> float:
        if len(base_chars) == 0:
            return 0.0
        best = 1
        current = 1
        for i in range(1, len(base_chars)):
            if base_chars[i] == base_chars[i - 1]:
                current += 1
                best = max(best, current)
            else:
                current = 1
        return float(best / len(base_chars))

    def _encode_single_motif_sequence(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.numel() == 0:
            return torch.zeros(
                self.hidden_dim,
                dtype=self.nt_embedding.weight.dtype,
                device=self.nt_embedding.weight.device,
            )

        token_ids = token_ids.long().to(self.nt_embedding.weight.device)
        emb = self.nt_embedding(token_ids)

        token_feat = self.seq_token_proj(emb)
        token_values = self.seq_value_proj(token_feat)
        token_gates = self.seq_attn_gate(token_feat).squeeze(-1)

        weights = torch.softmax(token_gates, dim=0).unsqueeze(-1)
        seq_emb = (token_values * weights).sum(dim=0)

        return seq_emb

    def _build_explicit_motif_attr(
        self,
        motif_token_ids_list,
        motif_type_ids: torch.Tensor,
        motif_lengths: torch.Tensor,
        motif_pairing_ratios: torch.Tensor,
        motif_gc_contents: torch.Tensor,
        motif_edge_side_ids: torch.Tensor,
        motif_is_terminal: torch.Tensor,
        motif_center_positions: torch.Tensor,
        motif_is_dna: torch.Tensor,
    ) -> torch.Tensor:
        device = motif_lengths.device
        n_motifs = motif_lengths.shape[0]

        motif_type_oh = F.one_hot(
            motif_type_ids.long().clamp(min=0, max=self.motif_type_vocab_size - 1),
            num_classes=self.motif_type_vocab_size,
        ).float()

        edge_side_oh = F.one_hot(
            motif_edge_side_ids.long().clamp(min=0, max=self.edge_side_vocab_size - 1),
            num_classes=self.edge_side_vocab_size,
        ).float()

        scaled_len = torch.log1p(motif_lengths.float()).unsqueeze(-1) / 5.0
        pairing = motif_pairing_ratios.float().unsqueeze(-1)
        gc = motif_gc_contents.float().unsqueeze(-1)
        terminal = motif_is_terminal.float().unsqueeze(-1)
        center_pos = motif_center_positions.float().unsqueeze(-1)

        frac_a_list = []
        frac_c_list = []
        frac_g_list = []
        frac_u_list = []
        frac_purine_list = []
        nt_entropy_list = []
        max_nt_fraction_list = []
        max_run_fraction_list = []
        flexibility_proxy_list = []
        wobble_pair_fraction_list = []

        for token_ids, pair_ratio, gc_content in zip(
            motif_token_ids_list,
            motif_pairing_ratios,
            motif_gc_contents,
        ):
            chars = self._token_ids_to_base_chars(token_ids)

            frac_a = self._nt_fraction(chars, "A")
            frac_c = self._nt_fraction(chars, "C")
            frac_g = self._nt_fraction(chars, "G")
            frac_u = self._nt_fraction(chars, "U")
            frac_purine = frac_a + frac_g
            entropy = self._nt_entropy(frac_a, frac_c, frac_g, frac_u)
            max_nt_fraction = max(frac_a, frac_c, frac_g, frac_u)
            max_run_fraction = self._max_run_fraction_from_chars(chars)

            flexibility_proxy = 0.5 * (1.0 - float(pair_ratio)) + 0.5 * (1.0 - float(gc_content))

            # Without local dot-bracket in the encoder, keep wobble proxy as 0
            # for on-the-fly fallback. Precomputed motif_attr contains the
            # real motif-local wobble fraction.
            wobble_pair_fraction = 0.0

            frac_a_list.append(frac_a)
            frac_c_list.append(frac_c)
            frac_g_list.append(frac_g)
            frac_u_list.append(frac_u)
            frac_purine_list.append(frac_purine)
            nt_entropy_list.append(entropy)
            max_nt_fraction_list.append(max_nt_fraction)
            max_run_fraction_list.append(max_run_fraction)
            flexibility_proxy_list.append(flexibility_proxy)
            wobble_pair_fraction_list.append(wobble_pair_fraction)

        frac_a = torch.tensor(frac_a_list, dtype=torch.float32, device=device).unsqueeze(-1)
        frac_c = torch.tensor(frac_c_list, dtype=torch.float32, device=device).unsqueeze(-1)
        frac_g = torch.tensor(frac_g_list, dtype=torch.float32, device=device).unsqueeze(-1)
        frac_u = torch.tensor(frac_u_list, dtype=torch.float32, device=device).unsqueeze(-1)
        frac_purine = torch.tensor(frac_purine_list, dtype=torch.float32, device=device).unsqueeze(-1)
        nt_entropy = torch.tensor(nt_entropy_list, dtype=torch.float32, device=device).unsqueeze(-1)
        max_nt_fraction = torch.tensor(max_nt_fraction_list, dtype=torch.float32, device=device).unsqueeze(-1)
        max_run_fraction = torch.tensor(max_run_fraction_list, dtype=torch.float32, device=device).unsqueeze(-1)
        flexibility_proxy = torch.tensor(flexibility_proxy_list, dtype=torch.float32, device=device).unsqueeze(-1)
        wobble_pair_fraction = torch.tensor(wobble_pair_fraction_list, dtype=torch.float32, device=device).unsqueeze(-1)
        is_dna = motif_is_dna.float().unsqueeze(-1)

        motif_attr = torch.cat(
            [
                motif_type_oh,
                edge_side_oh,
                scaled_len,
                pairing,
                gc,
                terminal,
                center_pos,
                frac_a,
                frac_c,
                frac_g,
                frac_u,
                frac_purine,
                nt_entropy,
                max_nt_fraction,
                max_run_fraction,
                flexibility_proxy,
                wobble_pair_fraction,
                is_dna,
            ],
            dim=-1,
        )

        if motif_attr.shape[-1] != self.motif_attr_dim:
            raise RuntimeError(
                f"motif_attr should be {self.motif_attr_dim}D, got {motif_attr.shape}"
            )

        return motif_attr

    def forward(
        self,
        motif_token_ids_list,
        motif_type_ids: torch.Tensor,
        motif_lengths: torch.Tensor,
        motif_pairing_ratios: torch.Tensor,
        motif_gc_contents: torch.Tensor,
        motif_edge_side_ids: torch.Tensor,
        motif_is_terminal: torch.Tensor,
        motif_center_positions: torch.Tensor,
        motif_is_dna: Optional[torch.Tensor] = None,
        precomputed_motif_attr: Optional[torch.Tensor] = None,
    ):
        motif_seq_embs = []

        for token_ids in motif_token_ids_list:
            seq_emb = self._encode_single_motif_sequence(token_ids)
            motif_seq_embs.append(seq_emb)

        if len(motif_seq_embs) == 0:
            raise ValueError("MotifEncoder received an empty motif_token_ids_list.")

        motif_seq_emb = torch.stack(motif_seq_embs, dim=0)

        motif_type_ids = motif_type_ids.long().to(motif_seq_emb.device)
        motif_edge_side_ids = motif_edge_side_ids.long().to(motif_seq_emb.device)
        motif_lengths = motif_lengths.float().to(motif_seq_emb.device)
        motif_pairing_ratios = motif_pairing_ratios.float().to(motif_seq_emb.device)
        motif_gc_contents = motif_gc_contents.float().to(motif_seq_emb.device)
        motif_is_terminal = motif_is_terminal.float().to(motif_seq_emb.device)
        motif_center_positions = motif_center_positions.float().to(motif_seq_emb.device)

        motif_type_emb = self.type_embedding(motif_type_ids)
        motif_edge_side_emb = self.edge_side_embedding(motif_edge_side_ids)

        motif_len_emb = self.length_proj(motif_lengths.unsqueeze(-1))
        motif_pair_emb = self.pairing_proj(motif_pairing_ratios.unsqueeze(-1))
        motif_gc_emb = self.gc_proj(motif_gc_contents.unsqueeze(-1))
        motif_terminal_emb = self.terminal_proj(motif_is_terminal.unsqueeze(-1))
        motif_center_pos_emb = self.center_pos_proj(motif_center_positions.unsqueeze(-1))

        summary_emb = self.summary_mlp(
            torch.cat(
                [
                    motif_type_emb,
                    motif_edge_side_emb,
                    motif_len_emb,
                    motif_pair_emb,
                    motif_gc_emb,
                    motif_terminal_emb,
                    motif_center_pos_emb,
                ],
                dim=-1,
            )
        )

        z_motif = self.fusion_mlp(
            torch.cat(
                [
                    motif_seq_emb,
                    summary_emb,
                    motif_type_emb,
                ],
                dim=-1,
            )
        )

        if precomputed_motif_attr is not None:
            motif_summary_feat = precomputed_motif_attr.float().to(z_motif.device)
        else:
            if motif_is_dna is None:
                motif_is_dna = torch.zeros_like(motif_lengths, dtype=torch.float32)
            else:
                motif_is_dna = motif_is_dna.float().to(z_motif.device)

            motif_summary_feat = self._build_explicit_motif_attr(
                motif_token_ids_list=motif_token_ids_list,
                motif_type_ids=motif_type_ids,
                motif_lengths=motif_lengths,
                motif_pairing_ratios=motif_pairing_ratios,
                motif_gc_contents=motif_gc_contents,
                motif_edge_side_ids=motif_edge_side_ids,
                motif_is_terminal=motif_is_terminal,
                motif_center_positions=motif_center_positions,
                motif_is_dna=motif_is_dna,
            )

        if motif_summary_feat.shape[-1] != self.motif_attr_dim:
            raise RuntimeError(
                f"precomputed_motif_attr should be {self.motif_attr_dim}D, "
                f"got {motif_summary_feat.shape}"
            )

        return {
            "z_motif": z_motif,
            "motif_attr": motif_summary_feat,
            "motif_summary_feat": motif_summary_feat,
        }