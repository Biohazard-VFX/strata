import torch


def composite_tensor(alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    trans = torch.cat((torch.ones_like(alpha[:, :1]), (1 - alpha[:, :-1]).cumprod(dim=1)), dim=1)
    return (trans[:, :, None] * rgb).sum(dim=1)


def deep_loss(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    mixed_weight: float = 12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weight = 1 + (mixed_weight - 1) * target["mixed"].float()
    valid = (target["inverse_depth"] > 0) & (target["alpha"] > 1e-7)
    depth_error = (
        prediction["inverse_depth"].clamp_min(1e-8).log()
        - target["inverse_depth"].clamp_min(1e-8).log()
    ).abs()
    depth = (depth_error * valid).sum() / valid.sum().clamp_min(1)
    gradient = gradient_loss(prediction["inverse_depth"], target["inverse_depth"], valid)
    alpha = ((prediction["alpha"] - target["alpha"]).abs() * weight[:, None]).sum() / (
        weight.sum() * target["alpha"].shape[1]
    ).clamp_min(1)
    rgb = ((prediction["rgb"] - target["rgb"]).abs() * weight[:, None, None]).sum() / (
        weight.sum() * target["rgb"].shape[1] * 3
    ).clamp_min(1)
    reconstruction = (
        (composite_tensor(prediction["alpha"], prediction["rgb"]) - target["beauty"]).abs()
        * weight[:, None]
    ).sum() / (weight.sum() * 3).clamp_min(1)
    terms = {
        "depth": depth,
        "gradient": gradient,
        "alpha": alpha,
        "rgb": rgb,
        "composite": reconstruction,
    }
    return depth + gradient + alpha + 0.25 * rgb + 0.25 * reconstruction, terms


def gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    total = prediction.new_zeros(())
    for stride in (1, 2, 4):
        residual = (prediction - target)[..., ::stride, ::stride]
        mask = valid[..., ::stride, ::stride]
        horizontal = mask[..., :, 1:] & mask[..., :, :-1]
        vertical = mask[..., 1:, :] & mask[..., :-1, :]
        errors = (residual[..., :, 1:] - residual[..., :, :-1]).abs() * horizontal
        errors_y = (residual[..., 1:, :] - residual[..., :-1, :]).abs() * vertical
        total = total + (errors.sum() + errors_y.sum()) / (
            horizontal.sum() + vertical.sum()
        ).clamp_min(1)
    return total / 3
