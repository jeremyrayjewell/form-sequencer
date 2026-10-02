"""Form Sequencer: an audio-driven generative form assembled from document fragments."""

import os
from pathlib import Path

# librosa JIT-compiles with numba, whose default cache lives next to the installed package.
# On system-wide installs (e.g. C:\Python313) that folder is read-only and numba spins
# retrying temp files, so point it at a per-user cache before librosa is ever imported.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path.home() / ".cache" / "form-sequencer" / "numba"))

__version__ = "0.1.0"
