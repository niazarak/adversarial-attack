import torch
from torch import nn


def clip(x, x_, eps):
    mask = torch.ones_like(x)
    lower_clip = torch.max(torch.stack([mask * 0, x - eps, x_]), dim=0)[0]
    return torch.min(torch.stack([mask, x + eps, lower_clip]), dim=0)[0]


def train_adv_examples(
        model: nn.Module, loss_fct: callable, adv_examples: torch.Tensor, adv_targets: torch.Tensor,
        epochs: int = 10, alpha: float = 1.0, clip_eps: float = (1 / 255) * 8, do_clip: bool = False, minimize: bool = False
):
    model.eval()

    for e in range(epochs):
        adv_examples.requires_grad = True
        model.zero_grad()

        adv_out = model(adv_examples)
        loss = loss_fct(adv_out, adv_targets)
        loss.backward()

        adv_grad = adv_examples.grad
        adv_examples = adv_examples.detach()

        direction = -1 if minimize else 1
        adv_sign_grad = adv_examples + direction * alpha * adv_grad.sign()

        if do_clip:
            adv_examples = clip(adv_examples, adv_sign_grad, clip_eps)
        else:
            adv_examples = adv_sign_grad

    return adv_examples


def train_adv_fgsm(
        model: nn.Module, loss_fct: callable, adv_examples: torch.Tensor, adv_targets: torch.Tensor,
        epochs: int = 10, alpha: float = 0.1
):
    return train_adv_examples(
        model, loss_fct, adv_examples, adv_targets,
        epochs=epochs, alpha=alpha, do_clip=False, minimize=False
    )


def train_adv_bim(
        model: nn.Module, loss_fct: callable, adv_examples: torch.Tensor, adv_targets: torch.Tensor,
        epochs: int = 10, alpha: float = 1.0, clip_eps: float = (1 / 255) * 8
):
    return train_adv_examples(
        model, loss_fct, adv_examples, adv_targets,
        epochs=epochs, alpha=alpha, do_clip=True, clip_eps=clip_eps, minimize=False
    )


def train_adv_least_likely(
        model: nn.Module, loss_fct: callable, adv_examples: torch.Tensor,
        epochs: int = 10, alpha: float = 0.1, clip_eps: float = (1 / 255) * 8
):
    model.eval()
    adv_targets = model(adv_examples).argmin(dim=1).detach()
    return train_adv_examples(
        model, loss_fct, adv_examples, adv_targets,
        epochs=epochs, alpha=alpha, do_clip=True, clip_eps=clip_eps, minimize=True
    )
