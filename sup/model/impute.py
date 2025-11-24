from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..catsample import sample_categorical
from ..graph_lib import Graph
from . import utils as mutils

Tensor = torch.Tensor


def impute_block_uniform(
    model: nn.Module,
    graph: Graph,
    noise: nn.Module,
    tokens: Optional[Tensor] = None,
    token_mask: Optional[Tensor] = None,
    numeric: Optional[Tensor] = None,
    numeric_mask: Optional[Tensor] = None,
    steps: int = 200,
    eps: float = 1e-5,
    denoise: bool = True,
    device: Optional[torch.device] = None,
    numeric_clip: Optional[float] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Impute missing values assuming a block-uniform diffusion graph (no absorbing state).
    """
    if getattr(graph, "absorb", False):
        raise ValueError("Expected a block-uniform graph without absorbing states.")
    return _impute_block_graph(
        model=model,
        graph=graph,
        noise=noise,
        tokens=tokens,
        token_mask=token_mask,
        numeric=numeric,
        numeric_mask=numeric_mask,
        steps=steps,
        eps=eps,
        denoise=denoise,
        device=device,
        numeric_clip=numeric_clip,
    )


def impute_block_absorbing(
    model: nn.Module,
    graph: Graph,
    noise: nn.Module,
    tokens: Optional[Tensor] = None,
    token_mask: Optional[Tensor] = None,
    numeric: Optional[Tensor] = None,
    numeric_mask: Optional[Tensor] = None,
    steps: int = 200,
    eps: float = 1e-5,
    denoise: bool = True,
    device: Optional[torch.device] = None,
    numeric_clip: Optional[float] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Impute missing values assuming a block-absorbing diffusion graph where each feature
    has an explicit mask state. `token_mask` should be True where entries are missing.
    """
    if not getattr(graph, "absorb", False):
        raise ValueError("Expected a block-absorbing graph (graph.absorb must be True).")
    return _impute_block_graph(
        model=model,
        graph=graph,
        noise=noise,
        tokens=tokens,
        token_mask=token_mask,
        numeric=numeric,
        numeric_mask=numeric_mask,
        steps=steps,
        eps=eps,
        denoise=denoise,
        device=device,
        numeric_clip=numeric_clip,
    )


def _impute_block_graph(
    model: nn.Module,
    graph: Graph,
    noise: nn.Module,
    tokens: Optional[Tensor] = None,
    token_mask: Optional[Tensor] = None,
    numeric: Optional[Tensor] = None,
    numeric_mask: Optional[Tensor] = None,
    steps: int = 200,
    eps: float = 1e-5,
    denoise: bool = True,
    device: Optional[torch.device] = None,
    numeric_clip: Optional[float] = None,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """
    Impute missing categorical and numerical features by running the PC sampler while
    clamping observed entries at every diffusion step.

    Args:
        model: Trained Tab-SEDD model.
        graph: Block-uniform graph describing the categorical layout.
        noise: Noise schedule used during training (HybridNoise or Noise).
        tokens: Local categorical tokens for each row (shape [B, n_cat]).
        token_mask: Boolean mask with True where categorical values should be imputed.
        numeric: Numerical features (shape [B, n_num]).
        numeric_mask: Boolean mask with True where numerical values should be imputed.
        steps: Number of predictor steps to run.
        eps: Terminal time used for PC sampling (matches training).
        denoise: Whether to run the final denoising predictor step.
        device: Optional torch.device override.
        numeric_clip: Optional clamp value applied after each update.

    Returns:
        Tuple of (categorical_tokens, numeric_values) where categorical tokens are in
        local indexing (per column). Observed entries remain untouched.
    """
    if device is None:
        device = next(model.parameters()).device

    batch_size = _infer_batch_size(tokens, numeric)
    model.eval()

    has_categorical = graph.dim > 0
    if not has_categorical and tokens is not None and tokens.numel() > 0:
        raise ValueError("Graph has no categorical dimension but categorical tokens were provided.")

    observed_tokens = tokens.to(device).long() if tokens is not None and tokens.numel() > 0 else None
    token_mask_tensor = _prepare_mask(token_mask, observed_tokens, "token_mask")

    observed_numeric = numeric.to(device).float() if numeric is not None and numeric.numel() > 0 else None
    numeric_mask_tensor = _prepare_mask(numeric_mask, observed_numeric, "numeric_mask")

    if observed_tokens is not None and token_mask_tensor is None:
        raise ValueError("token_mask must be provided when categorical context is supplied.")
    if observed_numeric is not None and numeric_mask_tensor is None:
        raise ValueError("numeric_mask must be provided when numerical context is supplied.")
    if observed_tokens is None and observed_numeric is None:
        raise ValueError("At least one of `tokens` or `numeric` must be provided for imputation.")

    tokens_state: Optional[Tensor]
    if has_categorical:
        tokens_state = graph.sample_limit(batch_size).to(device)
        tokens_state = _apply_token_constraints(graph, tokens_state, observed_tokens, token_mask_tensor)
    else:
        tokens_state = None
        token_mask_tensor = None

    numeric_dim = observed_numeric.shape[1] if observed_numeric is not None else (
        max(getattr(model, "input_dim", graph.dim) - graph.dim, 0)
    )
    if numeric_dim <= 0:
        numeric_state: Optional[Tensor] = None
        observed_numeric = None
        numeric_mask_tensor = None
    else:
        init_sigma = _estimate_initial_sigma(noise, device, use_numeric=True)
        if observed_numeric is not None:
            random_init = torch.randn_like(observed_numeric) * init_sigma
            numeric_state = torch.where(numeric_mask_tensor, random_init, observed_numeric)
        else:
            numeric_state = torch.randn(batch_size, numeric_dim, device=device) * init_sigma
            observed_numeric = None
            numeric_mask_tensor = None
        if numeric_clip is not None and numeric_clip > 0:
            numeric_state = numeric_state.clamp_(-numeric_clip, numeric_clip)

    base_score_fn = mutils.get_score_fn(model, train=False, sampling=True, discrete_dim=graph.dim)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    for idx in range(steps):
        t_curr = timesteps[idx].expand(batch_size, 1)
        t_next = timesteps[idx + 1].expand(batch_size, 1)

        sigma_cat_curr, _ = noise(t_curr)
        sigma_cat_next, _ = noise(t_next)
        if hasattr(noise, "numeric_schedule"):
            sigma_num_curr, _ = noise.numeric_schedule(t_curr)
            sigma_num_next, _ = noise.numeric_schedule(t_next)
        else:
            sigma_num_curr, sigma_num_next = sigma_cat_curr, sigma_cat_next

        sigma_arg = sigma_cat_curr if has_categorical else sigma_num_curr
        tokens_state = _apply_token_constraints(graph, tokens_state, observed_tokens, token_mask_tensor)
        features = _build_features(graph, tokens_state, numeric_state, batch_size, device)
        scores = base_score_fn(features, sigma_arg)
        discrete_scores = scores[:, :graph.dim]
        numeric_scores = scores[:, graph.dim:] if numeric_state is not None else None

        if has_categorical and tokens_state is not None:
            dsigma_cat = sigma_cat_curr - sigma_cat_next
            tokens_state = _apply_token_constraints(graph, tokens_state, observed_tokens, token_mask_tensor)
            probs = graph.staggered_score(discrete_scores, dsigma_cat) * graph.transp_transition(tokens_state, dsigma_cat)
            tokens_state = sample_categorical(probs)
            tokens_state = _apply_token_constraints(graph, tokens_state, observed_tokens, token_mask_tensor)

        if numeric_state is not None and numeric_scores is not None:
            sigma_sq_curr = sigma_num_curr ** 2
            sigma_sq_next = sigma_num_next ** 2
            delta_sigma_sq = torch.clamp(sigma_sq_curr - sigma_sq_next, min=0.0)
            delta_sigma_sq = _match_numeric_shape(delta_sigma_sq, numeric_state)
            noise_scale = torch.sqrt(delta_sigma_sq + 1e-12)
            stochastic_term = noise_scale * torch.randn_like(numeric_state)
            numeric_state = numeric_state + delta_sigma_sq * numeric_scores + stochastic_term
            numeric_state = _apply_numeric_constraints(numeric_state, observed_numeric, numeric_mask_tensor)
            if numeric_clip is not None and numeric_clip > 0:
                numeric_state = numeric_state.clamp_(-numeric_clip, numeric_clip)

    if denoise:
        t_final = timesteps[-1].expand(batch_size, 1)
        sigma_cat_final, _ = noise(t_final)
        if hasattr(noise, "numeric_schedule"):
            sigma_num_final, _ = noise.numeric_schedule(t_final)
        else:
            sigma_num_final = sigma_cat_final

        sigma_arg = sigma_cat_final if has_categorical else sigma_num_final
        tokens_state = _apply_token_constraints(graph, tokens_state, observed_tokens, token_mask_tensor)
        features = _build_features(graph, tokens_state, numeric_state, batch_size, device)
        scores = base_score_fn(features, sigma_arg)
        discrete_scores = scores[:, :graph.dim]
        numeric_scores = scores[:, graph.dim:] if numeric_state is not None else None

        if has_categorical and tokens_state is not None:
            probs = graph.staggered_score(discrete_scores, sigma_cat_final) * graph.transp_transition(tokens_state, sigma_cat_final)
            tokens_state = sample_categorical(probs)
            tokens_state = _apply_token_constraints(graph, tokens_state, observed_tokens, token_mask_tensor)

        if numeric_state is not None and numeric_scores is not None:
            sigma_sq_final = _match_numeric_shape(sigma_num_final ** 2, numeric_state)
            numeric_state = numeric_state + sigma_sq_final * numeric_scores
            numeric_state = _apply_numeric_constraints(numeric_state, observed_numeric, numeric_mask_tensor)
            if numeric_clip is not None and numeric_clip > 0:
                numeric_state = numeric_state.clamp_(-numeric_clip, numeric_clip)

    categorical_local: Optional[Tensor] = None
    if tokens_state is not None:
        tokens_state = _apply_token_constraints(graph, tokens_state, observed_tokens, token_mask_tensor)
        categorical_local = graph.to_local(tokens_state)

    return categorical_local, numeric_state


def _infer_batch_size(tokens: Optional[Tensor], numeric: Optional[Tensor]) -> int:
    batch_size: Optional[int] = None
    if tokens is not None and tokens.numel() > 0:
        batch_size = tokens.shape[0]
    if numeric is not None and numeric.numel() > 0:
        if batch_size is None:
            batch_size = numeric.shape[0]
        elif numeric.shape[0] != batch_size:
            raise ValueError("Categorical and numerical batches must have the same size.")
    if batch_size is None:
        raise ValueError("Cannot infer batch size without categorical or numerical inputs.")
    return batch_size


def _prepare_mask(mask: Optional[Tensor], values: Optional[Tensor], name: str) -> Optional[Tensor]:
    if values is None or mask is None:
        return None if values is None else mask
    mask_tensor = mask.to(values.device)
    if mask_tensor.shape != values.shape:
        raise ValueError(f"{name} must have shape {values.shape}, got {mask_tensor.shape}.")
    return mask_tensor.bool()


def _apply_token_constraints(
    graph: Graph,
    tokens_global: Optional[Tensor],
    observed_local: Optional[Tensor],
    mask: Optional[Tensor],
) -> Optional[Tensor]:
    if tokens_global is None or observed_local is None or mask is None:
        return tokens_global
    local = graph.to_local(tokens_global).long()
    clamped = torch.where(mask, local, observed_local)
    return graph.to_global(clamped)


def _apply_numeric_constraints(
    current: Tensor,
    observed: Optional[Tensor],
    mask: Optional[Tensor],
) -> Tensor:
    if observed is None or mask is None:
        return current
    return torch.where(mask, current, observed)


def _build_features(
    graph: Graph,
    tokens_global: Optional[Tensor],
    numeric_state: Optional[Tensor],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    cat_features = _build_categorical_features(graph, tokens_global, batch_size, device)
    if numeric_state is None or numeric_state.numel() == 0:
        return cat_features
    if cat_features.numel() == 0:
        return numeric_state
    return torch.cat([numeric_state, cat_features], dim=-1)


def _build_categorical_features(
    graph: Graph,
    tokens_global: Optional[Tensor],
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if tokens_global is None or graph.dim == 0:
        return torch.zeros(batch_size, 0, device=device)
    try:
        group_sizes = list(graph.group_sizes)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise ValueError("Imputation requires a block-uniform graph with group_sizes metadata.") from exc

    local_tokens = graph.to_local(tokens_global).long()
    one_hots = [
        F.one_hot(local_tokens[:, idx].clamp(min=0, max=size - 1), num_classes=size).float()
        for idx, size in enumerate(group_sizes)
    ]
    if not one_hots:
        return torch.zeros(batch_size, 0, device=device)
    return torch.cat(one_hots, dim=-1)


def _match_numeric_shape(values: Tensor, reference: Tensor) -> Tensor:
    while values.ndim < reference.ndim:
        values = values.unsqueeze(-1)
    return values.expand_as(reference)


def _estimate_initial_sigma(noise: nn.Module, device: torch.device, use_numeric: bool) -> float:
    t = torch.ones(1, 1, device=device)
    if use_numeric and hasattr(noise, "numeric_schedule"):
        sigma = noise.numeric_schedule(t)[0]
    else:
        sigma = noise(t)[0]
    return float(sigma.max().item())
