"""Loader for the published CodeFormer checkpoint.

The release artifact at
``https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth``
is a torch.save dict of the form::

    {
      "params_ema": <state_dict>,  # recommended
      "params":     <state_dict>,  # fallback
      ...
    }

Both state dicts match :class:`CodeFormer` keys verbatim, so
``load_state_dict`` works without remapping.
"""

from __future__ import annotations

import logging
import os
from typing import Union

import torch

from .codeformer_arch import CodeFormer

logger = logging.getLogger(__name__)


# Upstream's v0.1.0 release. Keep this in sync with the README. We do
# NOT auto-download by default because the inference server should be
# deterministic at startup; pass ``download_if_missing=True`` for the
# one-off setup script.
DEFAULT_CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)


def load_codeformer(
    checkpoint_path: Union[str, os.PathLike],
    device: Union[str, torch.device] = "cpu",
    download_if_missing: bool = False,
    download_url: str = DEFAULT_CODEFORMER_URL,
) -> CodeFormer:
    """Build a CodeFormer model and load the released weights.

    Args:
        checkpoint_path: local path to ``codeformer.pth``. Relative paths
            are resolved by the caller -- we accept whatever string is
            given and let ``torch.load`` raise if the file is missing.
        device: device the model lives on after loading. The state dict
            is loaded onto CPU first to avoid GPU OOM during weight
            init (CodeFormer's transformer is ~1 GB).
        download_if_missing: when True, fetch the upstream
            ``codeformer.pth`` and store it at ``checkpoint_path`` if no
            file is present. Default ``False`` so the inference server
            never reaches the network at startup.
        download_url: URL to fetch from when
            ``download_if_missing=True``.

    Returns:
        A CodeFormer in ``eval()`` mode with ``requires_grad=False`` on
        every parameter. Callers can move it to GPU and call
        ``net(x, w=...)`` directly.
    """
    checkpoint_path = str(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        if not download_if_missing:
            raise FileNotFoundError(
                f"CodeFormer checkpoint not found at {checkpoint_path!r}. "
                "Place codeformer.pth there, or call "
                "load_codeformer(..., download_if_missing=True) once to fetch it."
            )
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)) or ".", exist_ok=True)
        logger.info("Downloading CodeFormer weights from %s to %s", download_url, checkpoint_path)
        torch.hub.download_url_to_file(download_url, checkpoint_path, progress=True)

    raw = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(raw, dict) and "params_ema" in raw:
        state_dict = raw["params_ema"]
    elif isinstance(raw, dict) and "params" in raw:
        state_dict = raw["params"]
    elif isinstance(raw, dict):
        # Already a bare state dict (some forks ship it that way).
        state_dict = raw
    else:
        raise ValueError(
            f"Unrecognized CodeFormer checkpoint format at {checkpoint_path!r}: "
            f"expected a dict with 'params_ema' or 'params' key, got {type(raw)}"
        )

    net = CodeFormer()
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    # The published checkpoint contains no extra keys. A few are
    # expected to be missing only if the user passed a half-trained
    # checkpoint; we still log them so it surfaces in API logs.
    if missing:
        logger.warning(
            "CodeFormer checkpoint is missing %d keys (first few: %s)",
            len(missing),
            missing[:5],
        )
    if unexpected:
        logger.warning(
            "CodeFormer checkpoint has %d unexpected keys (first few: %s)",
            len(unexpected),
            unexpected[:5],
        )

    net = net.to(device)
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    return net