# Fourier Holographic Reduced Representation (FHRR) encoding for FVSAIT
#
# Implements the hyper-vector algebra described in:
#   "FVSAIT: an Improved Image Translation Model using FHRR-based Encoding
#    and Lookahead Optimization"
#
#   * phase encoding      -- Eq.(1):  v_k = e^{i*theta_k}, theta_k = pi*(x_k+1)
#   * binding/unbinding   -- Eq.(2):  a (*) b = a . b   (element-wise complex
#                          mult);  unbind with the complex conjugate
#   * bundling            -- Eq.(3):  s = ph(sum_j v_j)  (sum + phase normalize)
#   * similarity          -- Eq.(4):  sim(a,b) = (1/m) sum_k Re(a_k b_k*)
#   * unbinding-by-retrieval -- Eq.(5) is evaluation-time only (not needed in
#                          training); provided for completeness.
#
# Hyper-vectors are carried as complex torch tensors (b, m, h, w); the phase
# representation (real-valued, rescaled to [-1, 1]) is what the mapping
# network R and the discriminator operate on, per the paper.

import torch


def phase_encode(x):
    """Encode real values in [-1, 1] to unit-magnitude complex hyper-vectors.

    Eq.(1):  theta_k = pi * (x_k + 1) in [0, 2*pi];
             v_k = e^{i*theta_k} = cos(theta_k) + i*sin(theta_k)

    Args:
        x: torch.Tensor with values in [-1, 1] (any shape)

    Returns:
        complex torch.Tensor of the same shape, |v_k| = 1
    """
    theta = (x + 1.0) * torch.pi
    return torch.complex(torch.cos(theta), torch.sin(theta))


def phase_decode(v, normalize=True):
    """Decode complex hyper-vectors to real values in [-1, 1].

    Inverse of `phase_encode` (up to the 2*pi periodicity): recover the
    principal angle and rescale to [-1, 1]. This is the input used by the
    mapping network R and the discriminator ("real-valued phase
    representations").

    Args:
        v: complex torch.Tensor
        normalize: True/False project to unit magnitude before decoding

    Returns:
        real torch.Tensor in [-1, 1] of the same shape
    """
    if normalize:
        v = v / (v.abs() + 1e-8)
    theta = torch.angle(v)  # in [-pi, pi]
    x = theta / torch.pi    # in [-1, 1]
    return x


def bind(a, b):
    """Binding: element-wise complex multiplication (Eq. 2, top)."""
    return a * b


def unbind(a, b):
    """Unbinding: multiply by the complex conjugate (Eq. 2, bottom).

    unbind(bind(a, b), b) == a
    """
    return a * b.conj()


def bundle(v_list):
    """Bundling: element-wise sum + phase normalization (Eq. 3).

    Args:
        v_list: list of complex hyper-vectors (same shape)

    Returns:
        ph(sum_j v_j) where ph(z) = z / |z|
    """
    s = torch.stack(v_list, dim=0).sum(dim=0)
    return s / (s.abs() + 1e-8)


def similarity(a, b, dim=1, keepdim=False):
    """Phase-based cosine similarity (Eq. 4).

    sim(a, b) = (1/m) * sum_k Re(a_k * conj(b_k)) = (1/m) sum_k cos(phi_a - phi_b)

    Args:
        a, b: complex torch tensors
        dim: dimension to average over (hypervector channels)

    Returns:
        real tensor with the similarity averaged over `dim`
    """
    return (a * b.conj()).real.mean(dim=dim, keepdim=keepdim)


def retrieve(s, candidates):
    """Unbundling by similarity-based retrieval (Eq. 5).

    Args:
        s: bundled complex hyper-vector
        candidates: list of hyper-vectors M

    Returns:
        the candidate maximizing sim(s, v)
    """
    sims = [similarity(s, v).mean() for v in candidates]
    return candidates[int(torch.tensor(sims).argmax())]
