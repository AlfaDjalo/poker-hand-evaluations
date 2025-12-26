import warnings
# Backwards-compatibility shim: prefer `from ML.models import ...` instead.
warnings.warn(
    "ML.models.implementation is deprecated — import from ML.models instead (e.g. `from ML.models import CardSetEncoder`). "
    "This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from . import *  # re-export everything from the package for compatibility