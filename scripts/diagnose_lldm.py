from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from double_attention.data import load_token_splits
from train_screen import get_batch, make_model


def attention_entropy_fraction(scores: torch.Tensor) -> float:
    """Mean causal attention entropy divided by its row-wise maximum."""

    length = scores.shape[-1]
    causal = torch.ones(length, length, device=scores.device, dtype=torch.bool).tril()
    probabilities = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    maximum = torch.arange(1, length + 1, device=scores.device).float().log()
    return float((entropy[..., 1:] / maximum[1:]).mean())


def query_statistics(query: torch.Tensor) -> dict[str, float]:
    """Statistics for query shaped [B, maps, T, R]."""

    normalized = F.normalize(query.float(), dim=-1)
    similarities = normalized @ normalized.transpose(-1, -2)
    length = query.shape[-2]
    off_diagonal = (
        similarities.sum(dim=(-1, -2))
        - similarities.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    ) / (length * (length - 1))

    effective_ranks: list[float] = []
    rank_fractions: list[float] = []
    for map_query in normalized.permute(1, 0, 2, 3):
        flattened = map_query.flatten(0, 1)
        centered = flattened - flattened.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        energy = singular_values.square()
        energy = energy / energy.sum().clamp_min(1e-12)
        effective_rank = float(torch.exp(-(energy * energy.clamp_min(1e-12).log()).sum()))
        effective_ranks.append(effective_rank)
        rank_fractions.append(effective_rank / min(centered.shape[0] - 1, centered.shape[1]))

    return {
        "token_q_cosine": float(off_diagonal.mean()),
        "q_effective_rank": sum(effective_ranks) / len(effective_ranks),
        "q_rank_fraction": sum(rank_fractions) / len(rank_fractions),
    }


def load_model(
    variant: str,
    checkpoint: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = make_model(variant, 2048, 64, "torch", 0, "tied").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    model.load_state_dict(state)
    return model.eval()


def gradient_statistics(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    groups: dict[str, tuple[str, ...]],
) -> dict[str, dict[str, float]]:
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(token_ids, targets)
    assert loss is not None
    loss.backward()

    result: dict[str, dict[str, float]] = {}
    named_parameters = dict(model.named_parameters())
    for group, patterns in groups.items():
        selected = [
            parameter
            for name, parameter in named_parameters.items()
            if any(pattern in name for pattern in patterns) and parameter.grad is not None
        ]
        count = sum(parameter.numel() for parameter in selected)
        if count == 0:
            continue
        weight_square = sum(float(parameter.detach().float().square().sum()) for parameter in selected)
        gradient_square = sum(float(parameter.grad.detach().float().square().sum()) for parameter in selected)
        weight_rms = math.sqrt(weight_square / count)
        gradient_rms = math.sqrt(gradient_square / count)
        result[group] = {
            "parameter_count": count,
            "weight_rms": weight_rms,
            "gradient_rms": gradient_rms,
            "gradient_weight_ratio": gradient_rms / max(weight_rms, 1e-12),
        }
    model.zero_grad(set_to_none=True)
    model.eval()
    return result


@torch.no_grad()
def diagnose_lldm(model: torch.nn.Module, hidden: torch.Tensor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for layer, block in enumerate(model.stack.blocks):
        features, assignments = block._feature_state(hidden)
        relational = assignments
        if block.config.center_relational_assignments:
            relational = assignments - assignments.mean(dim=-1, keepdim=True)
        if block.config.independent_relational_readouts:
            relational = assignments - assignments.mean(dim=-1, keepdim=True)
            queries = torch.stack(
                [
                    F.normalize(projection(relational), dim=-1, eps=block.config.eps)
                    for projection in block.query_projections
                ],
                dim=1,
            )
            keys = torch.stack(
                [
                    F.normalize(projection(relational), dim=-1, eps=block.config.eps)
                    for projection in block.key_projections
                ],
                dim=1,
            )
        else:
            queries = torch.stack(
                [
                    F.normalize(
                        projection(relational * block.query_gates[index]),
                        dim=-1,
                        eps=block.config.eps,
                    )
                    for index, projection in enumerate(block.relational_projections)
                ],
                dim=1,
            )
            keys = torch.stack(
                [
                    F.normalize(
                        projection(relational * block.key_gates[index]),
                        dim=-1,
                        eps=block.config.eps,
                    )
                    for index, projection in enumerate(block.relational_projections)
                ],
                dim=1,
            )
        score_scales = block.log_score_scales.exp()[None, :, None, None]
        scores = score_scales * (queries @ keys.transpose(-1, -2))
        centered = features - features.mean(dim=-1, keepdim=True)
        assignment_entropy = -(
            assignments * assignments.clamp_min(1e-12).log()
        ).sum(dim=-1).mean() / math.log(assignments.shape[-1])
        row = {
            "layer": layer,
            "attention_entropy_fraction": attention_entropy_fraction(scores),
            "assignment_entropy_fraction": float(assignment_entropy),
            **query_statistics(queries),
            "feature_rms": float(features.square().mean().sqrt()),
            "feature_centered_rms": float(centered.square().mean().sqrt()),
        }
        rows.append(row)
        hidden = block(hidden)
    return rows


@torch.no_grad()
def diagnose_a1(model: torch.nn.Module, hidden: torch.Tensor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    dictionaries = tuple(bank.normalized(model.stack.config.eps) for bank in model.stack.banks)
    for layer, block in enumerate(model.stack.blocks):
        bank_index = min(
            layer // model.stack.dictionary_group_size,
            len(model.stack.banks) - 1,
        )
        _, aux = block.attention(
            block.attention_norm(hidden),
            bank=model.stack.banks[bank_index],
            normalized_dictionary=dictionaries[bank_index],
            return_aux=True,
        )
        scores = aux.score_scales[None, :, None, None] * (
            aux.score_query @ aux.score_key.transpose(-1, -2)
        )
        rows.append(
            {
                "layer": layer,
                "attention_entropy_fraction": attention_entropy_fraction(scores),
                **query_statistics(aux.score_query),
            }
        )
        hidden = block(
            hidden,
            model.stack.banks[bank_index],
            normalized_dictionary=dictionaries[bank_index],
        )
    return rows


@torch.no_grad()
def diagnose_mha(model: torch.nn.Module, hidden: torch.Tensor) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for layer, block in enumerate(model.stack.blocks):
        normalized = block.attention_norm(hidden)
        batch, length, _ = normalized.shape
        attention = block.attention
        query = attention.query(normalized).view(
            batch, length, attention.heads, attention.head_dim
        ).transpose(1, 2)
        key = attention.key(normalized).view(
            batch, length, attention.heads, attention.head_dim
        ).transpose(1, 2)
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(attention.head_dim)
        rows.append(
            {
                "layer": layer,
                "attention_entropy_fraction": attention_entropy_fraction(scores),
                **query_statistics(query),
            }
        )
        hidden = block(hidden)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--lldm-variant", default="lldm2-pm")
    parser.add_argument("--lldm-checkpoint", type=Path, required=True)
    parser.add_argument("--a1-checkpoint", type=Path, required=True)
    parser.add_argument("--mha-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    training = load_token_splits(args.ids, None, None, None)[0]
    token_ids, targets = get_batch(
        training,
        torch.Generator().manual_seed(1701),
        8,
        64,
        device,
    )
    models = {
        "lldm": load_model(args.lldm_variant, args.lldm_checkpoint, device),
        "a1": load_model("a1", args.a1_checkpoint, device),
        "mha4": load_model("mha4", args.mha_checkpoint, device),
    }
    for name, model in models.items():
        positions = torch.arange(64, device=device)
        hidden = model.token_embedding(token_ids) + model.position_embedding(positions)[None]
        if name == "lldm":
            rows = diagnose_lldm(model, hidden)
        elif name == "a1":
            rows = diagnose_a1(model, hidden)
        else:
            rows = diagnose_mha(model, hidden)
        if name == "lldm":
            groups = {
                "dictionary": ("dictionary.weight",),
                "relational_projection": ("relational_projections",),
                "query_projection": ("query_projections",),
                "key_projection": ("key_projections",),
                "value_projection": ("value_projections",),
                "context_projection": ("context_projections",),
                "query_gate": ("query_gates",),
                "key_gate": ("key_gates",),
                "value_gate": ("value_gates",),
                "score_scale": ("log_score_scales",),
                "context_scale": ("context_scale",),
                "output": (".output.weight",),
            }
        else:
            groups = {
                "attention_query": ("attention.query.weight",),
                "attention_key": ("attention.key.weight",),
                "attention_value": ("attention.value.weight",),
                "attention_output": ("attention.output.weight",),
                "feedforward_input": ("feedforward.0.weight",),
                "feedforward_output": ("feedforward.2.weight",),
                "dictionary": ("raw_key", "raw_value"),
            }
        gradients = gradient_statistics(model, token_ids, targets, groups)
        print(json.dumps({"model": name, "layers": rows, "gradients": gradients}))


if __name__ == "__main__":
    main()
