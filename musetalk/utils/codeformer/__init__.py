"""Self-contained CodeFormer vendoring for inference.

Exposes the :class:`CodeFormer` model class and a small helper to
load the published ``codeformer.pth`` checkpoint. The vendored
package intentionally does **not** depend on ``basicsr``: importing
``basicsr`` pulls in a long tail of training-only dependencies
(``facexlib``, ``lmdb``, ``tb-nightly``, ...) that we don't need at
inference time. See ``vqgan_arch.py`` for the full rationale.

Typical use::

    from musetalk.utils.codeformer import load_codeformer
    net = load_codeformer("models/codeformer/codeformer.pth", device="cuda")
    with torch.no_grad():
        restored, _, _ = net(face_tensor, w=0.5, adain=True)
"""

from .codeformer_arch import CodeFormer
from .loader import load_codeformer

__all__ = ["CodeFormer", "load_codeformer"]