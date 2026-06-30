"""
µ-law codec utilities for non-AudioSocket telephony paths.

AudioSocket channels use SLIN (signed linear 16-bit, 8 kHz) natively, so these
functions are not needed for the primary AudioSocket flow.  They exist for future
paths that receive µ-law audio directly — e.g. raw RTP channels or legacy
FreePBX trunks that are not routed through the AudioSocket application.

All functions operate on raw bytes.  The sample width passed to audioop is always
2 (16-bit samples).  Resampling uses a stateless call (state=None) which is safe
for one-shot conversion of complete frames; callers that need to preserve
inter-frame state across a streaming session should maintain the returned state
tuple themselves and pass it back on subsequent calls.

Python 3.11 stdlib audioop is used.  Do NOT replace with audioop-lts — that
package targets Python 3.12+ where audioop was removed from the stdlib.
"""

import audioop


def ulaw_to_linear(data: bytes) -> bytes:
    return audioop.ulaw2lin(data, 2)


def linear_to_ulaw(data: bytes) -> bytes:
    return audioop.lin2ulaw(data, 2)


def resample_16k_to_8k(data: bytes) -> bytes:
    return audioop.ratecv(data, 2, 1, 16000, 8000, None)[0]


def resample_8k_to_16k(data: bytes) -> bytes:
    return audioop.ratecv(data, 2, 1, 8000, 16000, None)[0]
