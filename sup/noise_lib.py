import abc
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


def _build_noise(
    cfg,
    num_numerical: Optional[int] = None,
    num_categorical: Optional[int] = None,
):
    if cfg is None:
        return None
    noise_type = cfg.type.lower()
    if noise_type == 'geometric':
        return GeometricNoise(cfg.sigma_min, cfg.sigma_max)
    elif noise_type == 'loglinear':
        eps_max = getattr(cfg, "eps_max", getattr(cfg, "eps", 1e-3))
        eps_min = getattr(cfg, "eps_min", 1e-5)
        return LogLinearNoise(eps_max=eps_max, eps_min=eps_min)
    elif noise_type == 'loglinear_per_column':
        if num_categorical is None:
            raise ValueError("loglinear_per_column noise requires the number of categorical features.")
        eps_max = getattr(cfg, "eps_max", getattr(cfg, "eps", 1e-3))
        eps_min = getattr(cfg, "eps_min", 1e-5)
        k_init = getattr(cfg, "k_init", -6.0)
        k_offset = getattr(cfg, "k_offset", 1.0)
        return LogLinearNoise_PerColumn(
            num_categories=int(num_categorical),
            eps_max=eps_max,
            eps_min=eps_min,
            k_init=k_init,
            k_offset=k_offset,
        )
    elif noise_type == 'power_mean':
        return PowerMeanNoise(cfg.sigma_min, cfg.sigma_max, cfg.rho)
    elif noise_type == 'power_mean_per_column':
        if num_numerical is None:
            raise ValueError("power_mean_per_column noise requires the number of numerical features.")
        return PowerMeanNoise_PerColumn(
            num_numerical=num_numerical,
            sigma_min=cfg.sigma_min,
            sigma_max=cfg.sigma_max,
            rho_init=getattr(cfg, "rho_init", 1.0),
            rho_offset=getattr(cfg, "rho_offset", 2.0),
        )
    else:
        raise ValueError(f"{cfg.type} is not a valid noise")


def get_noise(
    config,
    numeric_config: Optional[object] = None,
    num_numerical: Optional[int] = None,
    num_categorical: Optional[int] = None,
):
    categorical_noise = _build_noise(config, num_categorical=num_categorical)
    numeric_noise = _build_noise(numeric_config, num_numerical=num_numerical)
    if categorical_noise is None:
        raise ValueError("Categorical noise configuration must be provided.")
    if numeric_noise is None:
        return categorical_noise
    return HybridNoise(categorical_noise, numeric_noise)


class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """

    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise is \int_0^t g(t) dt + g(0)
        """
        pass


class GeometricNoise(Noise, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise, nn.Module):
    """Log Linear noise schedule."""

    def __init__(self, eps_max=1e-3, eps_min=1e-5, **kwargs):
        super().__init__()
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.sigma_max = self.total_noise(torch.tensor(1.0))
        self.sigma_min = self.total_noise(torch.tensor(0.0))

    def k(self):
        return torch.tensor(1)

    def rate_noise(self, t):
        return (1 - self.eps_max - self.eps_min) / (1 - ((1 - self.eps_max - self.eps_min) * t + self.eps_min))

    def total_noise(self, t):
        """
        sigma_min = -log(1-eps_min), when t=0
        sigma_max = -log(eps_max), when t = 1
        """
        return -torch.log1p(-((1 - self.eps_max - self.eps_min) * t + self.eps_min))


class PowerMeanNoise(Noise):
    """
    The noise schedule using the power mean interpolation function.

    This is the schedule used in EDM
    """

    def __init__(self, sigma_min=0.002, sigma_max=80, rho=7, **kwargs):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.raw_rho = rho

    def rho(self):
        # Return the softplus-transformed rho for all num_numerical values
        return torch.tensor(self.raw_rho)

    def total_noise(self, t):
        sigma = (self.sigma_min ** (1 / self.rho()) + t * (
                self.sigma_max ** (1 / self.rho()) - self.sigma_min ** (1 / self.rho()))).pow(self.rho())
        return sigma

    def rate_noise(self, t):
        rho = self.rho()
        sigma_min_pow = self.sigma_min ** (1 / rho)
        sigma_max_pow = self.sigma_max ** (1 / rho)
        base = sigma_min_pow + t * (sigma_max_pow - sigma_min_pow)
        return rho * base.pow(rho - 1) * (sigma_max_pow - sigma_min_pow)

    def inverse_to_t(self, sigma):
        """
        Inverse function to map sigma back to t, with proper broadcasting support.
        sigma: [batch_size, num_numerical] or [batch_size, 1]
        Returns: t: [batch_size, num_numerical]
        """
        rho = self.rho()

        sigma_min_pow = self.sigma_min ** (1 / rho)  # Shape: [num_numerical]
        sigma_max_pow = self.sigma_max ** (1 / rho)  # Shape: [num_numerical]

        # To enable broadcasting between sigma and the per-column rho values, expand rho where needed.
        t = (sigma.pow(1 / rho) - sigma_min_pow) / (sigma_max_pow - sigma_min_pow)

        return t


class HybridNoise(nn.Module):
    """
    Wrap two independent noise schedules: categorical (discrete) and numeric (continuous).
    When called directly, behaves like the categorical schedule to remain compatible with
    existing discrete diffusion code. Numeric schedule can be accessed through
    `numeric_schedule`.
    """

    def __init__(self, categorical_noise: Noise, numeric_noise: Noise):
        super().__init__()
        self.categorical_noise = categorical_noise
        self.numeric_noise = numeric_noise

    def forward(self, t):
        return self.categorical_noise(t)

    def categorical_schedule(self, t):
        return self.categorical_noise(t)

    def numeric_schedule(self, t):
        return self.numeric_noise(t)


class PowerMeanNoise_PerColumn(Noise):

    def __init__(
            self,
            num_numerical: int,
            sigma_min: float = 0.002,
            sigma_max: float = 80,
            rho_init: float = 1.0,
            rho_offset: float = 2.0,
            **kwargs,
    ):
        super().__init__()
        if num_numerical <= 0:
            raise ValueError("num_numerical must be positive for per-column power mean noise.")
        self.num_numerical = int(num_numerical)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho_offset = float(rho_offset)
        self.rho_raw = nn.Parameter(
            torch.full((self.num_numerical,), float(rho_init), dtype=torch.float32)
        )

    def rho(self) -> torch.Tensor:
        # Return the softplus-transformed rho for all num_numerical values
        return nn.functional.softplus(self.rho_raw) + self.rho_offset

    def _prep_t(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.rho_raw.device, dtype=self.rho_raw.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return t

    def _sigma_bounds(
            self, device: torch.device, rho: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma_min = torch.tensor(self.sigma_min, device=device, dtype=rho.dtype)
        sigma_max = torch.tensor(self.sigma_max, device=device, dtype=rho.dtype)
        sigma_min_pow = torch.pow(sigma_min, 1 / rho)
        sigma_max_pow = torch.pow(sigma_max, 1 / rho)
        return sigma_min_pow, sigma_max_pow

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        t = self._prep_t(t)
        rho = self.rho().to(t.device)
        sigma_min_pow, sigma_max_pow = self._sigma_bounds(t.device, rho)
        base = sigma_min_pow + t * (sigma_max_pow - sigma_min_pow)
        return base.pow(rho)

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        t = self._prep_t(t)
        rho = self.rho().to(t.device)
        sigma_min_pow, sigma_max_pow = self._sigma_bounds(t.device, rho)
        base = sigma_min_pow + t * (sigma_max_pow - sigma_min_pow)
        return rho * base.pow(rho - 1) * (sigma_max_pow - sigma_min_pow)

    def inverse_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        rho = self.rho().to(sigma.device)
        sigma = sigma.to(rho.device)
        sigma_min_pow, sigma_max_pow = self._sigma_bounds(sigma.device, rho)
        return (sigma.pow(1 / rho) - sigma_min_pow) / (sigma_max_pow - sigma_min_pow)


class LogLinearNoise_PerColumn(Noise):
    def __init__(
        self,
        num_categories: int,
        eps_max: float = 1e-3,
        eps_min: float = 1e-5,
        k_init: float = -6.0,
        k_offset: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        if num_categories <= 0:
            raise ValueError("num_categories must be positive for loglinear_per_column noise.")
        self.num_categories = int(num_categories)
        self.eps_max = float(eps_max)
        self.eps_min = float(eps_min)
        self.k_offset = float(k_offset)
        self.k_raw = nn.Parameter(torch.full((self.num_categories,), float(k_init), dtype=torch.float32))

    def k(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.k_raw) + self.k_offset

    def _prep_t(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(self.k_raw.device, dtype=self.k_raw.dtype)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return t

    def _broadcast_k(self, t: torch.Tensor) -> torch.Tensor:
        k = self.k().to(t.device)
        while k.ndim < t.ndim:
            k = k.unsqueeze(0)
        return k

    def rate_noise(self, t: torch.Tensor):
        t = self._prep_t(t)
        k = self._broadcast_k(t)
        numerator = (1 - self.eps_max - self.eps_min) * k * torch.pow(t, k - 1)
        denominator = 1 - ((1 - self.eps_max - self.eps_min) * torch.pow(t, k) + self.eps_min)
        return numerator / denominator

    def total_noise(self, t: torch.Tensor):
        t = self._prep_t(t)
        k = self._broadcast_k(t)
        inner = (1 - self.eps_max - self.eps_min) * torch.pow(t, k) + self.eps_min
        return -torch.log1p(-inner)
