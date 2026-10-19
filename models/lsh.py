# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from . import utils


class FHRR(torch.nn.Module):
    """
    Fourier Holographic Reduced Representation (FHRR) phase encoding

    Project a (patch-normalized) feature vector to a complex hypervector
    z_d = exp(i * theta_d), where the phases theta are random projections of the
    (unit-normalized) input:

        theta_d = pi * <R_d, x_patch>,   R_d ~ unit-normalized random vector

    The phases (a real tensor) are returned; the complex hypervector is
    exp(i * theta). Binding of two hypervectors corresponds to phase addition
    (see `vsa_bind`), and similarity to the mean of cos(theta_1 - theta_2)
    (see `vsa_similarity`).

    Args:
        in_channels: number of input channels
        out_channels: hypervector dimension m (number of phases)
        kernel_size: kernel size for patch normalization and projection
        **kwargs: keyword arguments passed to `torch.conv2d` (e.g., stride)

    Note:
        Input features are normalized with respect to the convolution kernel so
        that the phases only depend on the direction of each patch, mirroring
        the normalization used by the MAP-B / LSH encoder below.
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = F._pair(kernel_size)

        w = torch.randn(out_channels, in_channels, *self.kernel_size)
        w = F.normalize(w, dim=[1, 2, 3])
        self.register_buffer("weight", w)

        self.conv_kwargs = kwargs

    def forward(self, x):
        """
        Encode input features to FHRR phases

        Args:
            x: torch.Tensor input (b, in_channels, h, w)

        Returns:
            phases theta (b, out_channels, h', w')
        """
        h, w = x.shape[2:]
        x = F.unfold(x, self.kernel_size, **self.conv_kwargs)
        x = F.normalize(x, dim=1)
        x = F.fold(x, (h, w), self.kernel_size, **self.conv_kwargs)
        return math.pi * torch.conv2d(x, self.weight, **self.conv_kwargs)


class FHRRBind(torch.nn.ModuleList):
    """
    Multi-scale FHRR encoder with binding across scales

    Each input scale is encoded to phases by a separate `FHRR` module, spatially
    aligned, and bound (phase addition) with per-scale normalization weights
    w_s = 1/S (paper Eq. 2).

    Args:
        in_channels: list of number of input channels per scale
        out_channels: hypervector dimension m
        kernel_size: kernel size per scale (scalar or list)
        **kwargs: per-scale keyword arguments passed to `FHRR` (e.g., stride)
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels

        n = len(in_channels)
        kernel_size = utils.parse_list_var(kernel_size, n)
        stride = utils.parse_list_var(kwargs.pop("stride", 1), n)
        for ch, k, s in zip(in_channels, kernel_size, stride):
            self.append(FHRR(ch, out_channels, kernel_size=k, stride=s))

    def forward(self, x_list):
        """
        Encode and bind a list of feature maps (phases bind by addition)

        Args:
            x_list: list of tensors (one per scale), or a single tensor
        """
        phases = None
        n = len(self)
        for i, fn in enumerate(self):
            x = x_list[i] if isinstance(x_list, list) else x_list
            p = fn(x)
            if phases is None:
                phases = p / n
            else:
                phases = phases + F.interpolate(
                    p, size=phases.shape[2:], mode="nearest"
                ) / n
        return phases


def vsa_bind(a, b, encoder="map"):
    """
    Bind two hypervectors.

    - 'map'  (MAP-B / LSH, real hypervectors): binding = elementwise product
    - 'fhrr' (phase hypervectors): binding = phase addition
    """
    if encoder == "fhrr":
        return a + b
    return a * b


def vsa_unbind(a, b, encoder="fhrr"):
    """
    Unbind a composed hypervector (exact inverse of `vsa_bind`).

    - 'fhrr': unbinding = phase subtraction, c = a - b, i.e. the complex
      product c = a * conj(b). Removing the predicted style phases from the
      generated hypervector maps it back to the source domain (paper Eq. 5).
    - 'map': no exact multiplicative inverse exists for real hypervectors;
      the elementwise product is returned for interface compatibility (the
      MAP-B ablation keeps the original binding-based training flow).
    """
    if encoder == "fhrr":
        return a - b
    return a * b


def vsa_similarity(a, b, encoder="map"):
    """
    Similarity between two bound hypervectors.

    - 'map'  : mean cosine similarity across channels
    - 'fhrr' : mean of cos(theta_a - theta_b), i.e. the real part of the
               normalized complex inner product Re<z_a, z_b>/m
    """
    if encoder == "fhrr":
        return torch.cos(a - b).mean()
    return F.cosine_similarity(a, b, dim=1).mean()


class LSH(torch.nn.Module):
    """
    Locality-Sensitive Hashing

    Project a feature vector or patch of feature vectors to a normalized
    space [-1,1] using a random weight matrix R containing unit-normalized rows

    Args:
        in_channels: number of input channels
        out_channels: number of output channels
        kernel_size: kernel size for convolution (input is unit normalized wrt kernel size)
        concatenate: True/False store sum of squares rather than normalize. Specifically
            used for `LSHCat` to concatenate outputs together. [default: False]
        **kwargs: keyword arguments passed to `torch.conv2d`

    Note:
        Input features are normalized with respect to the convolution kernel such that
        the output vectors are the cosine similarity with each row in the random matrix R.
        The weight matrix R is not learned, but this approach is useful for projecting
        features into a normalized space for vector symbolic architectures (VSAs).
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, concatenate=False, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = F._pair(kernel_size)
        self.concatenate = concatenate

        w = torch.randn(out_channels, in_channels, *self.kernel_size)
        if self.concatenate:
            self.register_buffer(
                "ssq_weight", torch.sum(w**2, dim=[1, 2, 3]).view(1, -1, 1, 1)
            )
            self.register_buffer("_ssq_conv", torch.ones(1, 1, *self.kernel_size))
        else:
            w = F.normalize(w, dim=[1, 2, 3])

        self.register_buffer("weight", w)

        self.conv_kwargs = kwargs

    def forward(self, x):
        """
        Apply LSH random projection

        Args:
            x: torch.Tensor input
        """
        # store ssq_x if concatenating
        if self.concatenate:
            self.ssq_x = torch.conv2d(
                torch.sum(x**2, dim=1, keepdim=True),
                self._ssq_conv,
                **self.conv_kwargs,
            )
        else:
            # normalize x
            h, w = x.shape[2:]
            x = F.unfold(x, self.kernel_size, **self.conv_kwargs)
            x = F.normalize(x, dim=1)
            x = F.fold(x, (h, w), self.kernel_size, **self.conv_kwargs)
        # convolve with normalized weight
        return torch.conv2d(x, self.weight, **self.conv_kwargs)


class LSHCat(torch.nn.ModuleList):
    """
    Locality-Sensitive Hashing with concatenated inputs

    Project a list of inputs by concatenating the input vectors for LSH

    Args:
        in_channels: list of number of input channels per input
        out_channels: number of output channels
        kernel_size: kernel size for convolution (input is unit normalized wrt kernel size)
            if list, kernel size per input
        **kwargs: keyword arguments passed to `torch.conv2d`. similar to `kernel_size`
            each kwarg value can also be a list of values per input

    Note:
        Separate LSH networks are initialized for each input but are applied as if
        inputs and projection weights were concatenated together for computing the
        cosine similarity (i.e., inputs and weights are normalized wrt all vectors).
    """

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels

        kwargs.update(
            {
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "concatenate": True,
            }
        )

        for i, ch in enumerate(self.in_channels):
            self.append(LSH(ch, **utils.parse_dict_var(kwargs, i)))

    def _apply_lsh(self, fn, x):
        # update w_norm
        w_norm = getattr(self, "w_norm", 0.0)
        w_norm += fn.ssq_weight
        self.w_norm = w_norm

        # apply lsh, get ssq_x
        x = fn(x)
        x_norm = fn.ssq_x

        # get x_out, x_norm
        x_out = getattr(self, "x_out", x)
        x_norm = getattr(self, "x_norm", x_norm)

        # update x_out/x_norm with resizing
        if hasattr(self, "x_out") and hasattr(self, "x_norm"):
            x_out = x_out + F.interpolate(x, size=x_out.shape[2:], mode="nearest")
            x_norm = x_norm + F.interpolate(
                fn.ssq_x, size=x_out.shape[2:], mode="nearest"
            )

        # update x_out, x_norm
        self.x_out = x_out
        self.x_norm = x_norm

    def _compute_total_cosine_similarity(self):
        # assert variables exist
        assert (
            hasattr(self, "x_out")
            and hasattr(self, "x_norm")
            and hasattr(self, "w_norm")
        ), "Missing stored variables. `_apply_lsh` must be called first."

        # compute total cosine similarity, reset variables
        out = self.x_out / (self.x_norm.sqrt() * self.w_norm.sqrt() + 1e-5)
        delattr(self, "x_out")
        delattr(self, "x_norm")
        delattr(self, "w_norm")
        return out

    def forward(self, x_list):
        """
        Concatenate and project list of tensors

        Args:
            x_list: list of tensors to concatenate and project
        """
        # apply each lsh
        for i, fn in enumerate(self):
            x = x_list[i] if isinstance(x_list, list) else x_list
            self._apply_lsh(fn, x)

        # compute total cosine similarity
        return self._compute_total_cosine_similarity()
