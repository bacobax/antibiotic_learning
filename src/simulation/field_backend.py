"""Utility helpers to run 2D grid operations on either NumPy/SciPy or Torch.

The biological simulation keeps most logic on the CPU through Mesa agents, but
large parts of the workload (diffusion, decay, convolutions) already operate on
dense grids.  This module provides a small abstraction that can execute those
array-wide kernels on a Torch device (CPU/GPU) when available, falling back to
SciPy for the legacy behaviour.  The implementation intentionally stays simple
and avoids forcing the rest of the codebase to store tensors instead of NumPy
arrays: callers pass in NumPy arrays and get NumPy arrays back, while the heavy
math can be offloaded to Torch internally when acceleration is enabled.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import convolve as scipy_convolve

try:  # Torch is optional for field acceleration
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch might not be installed in tests
    torch = None  # type: ignore
    F = None  # type: ignore
    TORCH_AVAILABLE = False


class FieldBackend:
    """Wrapper that executes grid operations on Torch when enabled."""

    def __init__(self, enabled: bool = False, device: Optional[str] = None):
        self.enabled = bool(enabled) and TORCH_AVAILABLE
        if self.enabled:
            requested_device = device if device is not None else "cuda"
            device_str = requested_device
            disable_accel = False
            if device_str.startswith("cuda") and not torch.cuda.is_available():  # type: ignore[attr-defined]
                if device is None:
                    disable_accel = True
                else:
                    device_str = "cpu"
            if disable_accel:
                self.enabled = False
                self.device = None
            else:
                try:
                    self.device = torch.device(device_str)
                except (TypeError, ValueError):  # Fallback to CPU
                    self.device = torch.device("cpu")
        else:
            self.device = None
        self._gaussian_kernel_cache: Dict[tuple[float, str], Any] = {}
        self._lap_kernel_cache: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def gaussian_filter(
        self,
        array: np.ndarray,
        sigma: float,
        mode: str = "constant",
        cval: float = 0.0,
    ) -> np.ndarray:
        """Apply Gaussian blur, using Torch when available.

        Args:
            array: Input NumPy array (modified copy is returned)
            sigma: Standard deviation of the Gaussian kernel
            mode: Padding mode ("constant" supported for Torch path)
            cval: Constant padding value when mode == "constant"
        """
        if sigma <= 0.0:
            return array

        if not self.enabled or F is None:
            return scipy_gaussian_filter(array, sigma=sigma, mode=mode, cval=cval)

        kernel = self._get_gaussian_kernel(sigma, mode)
        tensor = self._to_tensor(array)
        radius = kernel.shape[-1] // 2

        padded = self._pad(tensor, (radius, radius, 0, 0), mode, cval)
        blurred = F.conv2d(padded, kernel)
        padded = self._pad(blurred, (0, 0, radius, radius), mode, cval)
        blurred = F.conv2d(padded, kernel.transpose(-1, -2))
        return self._to_numpy(blurred)

    def convolve(
        self,
        array: np.ndarray,
        kernel: np.ndarray,
        mode: str = "nearest",
    ) -> np.ndarray:
        """Apply convolution with the provided kernel."""
        if not self.enabled or F is None:
            return scipy_convolve(array, kernel, mode=mode)

        tensor = self._to_tensor(array)
        k_tensor = self._get_laplacian_kernel(kernel)
        pad_y = (k_tensor.shape[-2] - 1) // 2
        pad_x = (k_tensor.shape[-1] - 1) // 2
        padded = self._pad(tensor, (pad_x, pad_x, pad_y, pad_y), mode, 0.0)
        result = F.conv2d(padded, k_tensor)
        return self._to_numpy(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_tensor(self, array: np.ndarray) -> Any:
        tensor = torch.as_tensor(array, dtype=torch.float32)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor.unsqueeze(0).unsqueeze(0)

    def _to_numpy(self, tensor: Any) -> np.ndarray:
        array = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
        return array

    def _pad(
        self,
        tensor: Any,
        padding: tuple[int, int, int, int],
        mode: str,
        cval: float,
    ) -> Any:
        if mode == "constant":
            return F.pad(tensor, padding, mode="constant", value=float(cval))
        # Approximate SciPy's "nearest" with replicate padding
        return F.pad(tensor, padding, mode="replicate")

    def _get_gaussian_kernel(self, sigma: float, mode: str) -> Any:
        key = (round(float(sigma), 3), mode)
        cached = self._gaussian_kernel_cache.get(key)
        if cached is not None:
            return cached
        radius = max(1, int(math.ceil(3.0 * max(1e-6, sigma))))
        size = radius * 2 + 1
        grid = torch.arange(size, dtype=torch.float32, device=self.device) - radius
        kernel_1d = torch.exp(-(grid ** 2) / (2.0 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()
        kernel_2d = kernel_1d.view(1, 1, 1, size)
        self._gaussian_kernel_cache[key] = kernel_2d
        return kernel_2d

    def _get_laplacian_kernel(self, kernel: np.ndarray) -> Any:
        key = kernel.size
        cached = self._lap_kernel_cache.get(key)
        if cached is not None:
            return cached
        tensor = torch.as_tensor(kernel, dtype=torch.float32)
        if self.device is not None:
            tensor = tensor.to(self.device)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        self._lap_kernel_cache[key] = tensor
        return tensor
