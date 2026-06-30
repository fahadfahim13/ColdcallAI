"""
Asterisk AudioSocket protocol handler — ColdCallAI telephony layer.

Protocol
--------
Every AudioSocket message uses a fixed 3-byte header followed by a variable-
length payload:

    ┌──────────┬───────────────┬───────────────┬──────────────────────┐
    │  type    │  length_hi    │  length_lo    │  payload …           │
    │  1 byte  │  1 byte       │  1 byte       │  length bytes        │
    └──────────┴───────────────┴───────────────┴──────────────────────┘

    length = (length_hi << 8) | length_lo

Message types
~~~~~~~~~~~~~
    0x00  HANGUP  — remote end has hung up.   length = 0, no payload.
    0x01  UUID    — Asterisk channel UUID.    payload = UTF-8 uuid string.
    0x10  AUDIO   — raw SLIN PCM.            payload = 320 bytes exactly
                    (160 samples × 2 bytes, signed-16-bit LE, 8 kHz mono, 20 ms).
    0x03  DTMF    — a key press.             payload = 1 ASCII byte.
    0xFF  ERROR   — Asterisk error report.   payload = UTF-8 message.

Connection handshake
~~~~~~~~~~~~~~~~~~~~
After TCP connect, Asterisk immediately sends a UUID (0x01) frame containing
the channel UUID.  All subsequent 0x10 frames carry inbound audio.  To return
synthesised speech, write identical 3-byte header + 320-byte SLIN payload back
over the same socket.

TCP_NODELAY — MANDATORY
~~~~~~~~~~~~~~~~~~~~~~~
AudioSocket streams 320-byte frames every 20 ms.  Without TCP_NODELAY the
kernel Nagle algorithm coalesces these small writes into larger TCP segments,
adding up to 200 ms of stall latency — completely unacceptable for real-time
voice.  Every accepted socket MUST be configured with:

    sock.setsockopt(IPPROTO_TCP, TCP_NODELAY, 1)

asyncio's stream writer has its own write-buffer that mirrors Nagle behaviour.
Disable it immediately after accept with:

    writer.transport.set_write_buffer_limits(0)
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

MSG_HANGUP: int = 0x00
MSG_UUID:   int = 0x01
MSG_AUDIO:  int = 0x10
MSG_DTMF:   int = 0x03
MSG_ERROR:  int = 0xFF

FRAME_BYTES: int = 320    # 20 ms of SLIN 8 kHz mono: 160 samples × 2 bytes
SAMPLE_RATE: int = 8_000  # Hz

# Drop outbound audio when the write buffer already contains more than 2 frames
# (40 ms) of unacknowledged data.  Anything larger means we are behind
# real-time — discarding prevents unbounded queue growth during barge-in.
_MAX_WRITE_BACKLOG: int = 2 * FRAME_BYTES  # 640 bytes


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AudioSocketError(Exception):
    """Base exception for AudioSocket protocol faults and I/O errors."""


class AudioSocketHangup(AudioSocketError):
    """Raised when Asterisk sends a HANGUP (0x00) frame or the TCP connection
    closes before a complete frame header can be read."""


# ---------------------------------------------------------------------------
# Low-level frame I/O
# ---------------------------------------------------------------------------

async def _read_frame(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    """Read exactly one AudioSocket frame from *reader*.

    Returns
    -------
    (msg_type, payload)
        *msg_type* is one of the ``MSG_*`` constants.  *payload* is the raw
        bytes that follow the 3-byte header (may be empty for zero-length
        frames).

    Raises
    ------
    AudioSocketHangup
        On a HANGUP (0x00) frame, or if the peer closes the connection while
        the 3-byte header is being read.
    AudioSocketError
        On an ERROR (0xFF) frame, or if the peer closes mid-payload.
    """
    try:
        header = await reader.readexactly(3)
    except asyncio.IncompleteReadError as exc:
        raise AudioSocketHangup(
            f"Connection closed during header read "
            f"(got {len(exc.partial)}/3 bytes)"
        ) from exc

    msg_type = header[0]
    length = (header[1] << 8) | header[2]

    if msg_type == MSG_HANGUP:
        raise AudioSocketHangup("Asterisk sent HANGUP (0x00) frame")

    if length == 0:
        return msg_type, b""

    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise AudioSocketError(
            f"Incomplete payload for type {hex(msg_type)}: "
            f"expected {length} bytes, received {len(exc.partial)}"
        ) from exc

    if msg_type == MSG_ERROR:
        raise AudioSocketError(
            f"Asterisk error frame: {payload.decode('utf-8', errors='replace')}"
        )

    return msg_type, payload


def _build_frame(msg_type: int, payload: bytes) -> bytes:
    """Pack *msg_type* and *payload* into a single :class:`bytes` object
    with the correct 3-byte AudioSocket header."""
    length = len(payload)
    return bytes([msg_type, (length >> 8) & 0xFF, length & 0xFF]) + payload


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class AudioSocketSession:
    """Represents one live AudioSocket call leg.

    Instantiated by :class:`AudioSocketServer` immediately after the UUID
    handshake completes.  Pass the session to your call-handling coroutine.

    * Read inbound PCM with :meth:`read_audio`.
    * Push synthesised speech back to Asterisk with :meth:`send_audio`.
    * Call :meth:`close` (or let :class:`AudioSocketServer` do it) when done.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        call_uuid: str,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._call_uuid = call_uuid

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def call_uuid(self) -> str:
        """Asterisk channel UUID received during the handshake."""
        return self._call_uuid

    @property
    def remote_addr(self) -> str:
        """Human-readable remote address, e.g. ``'10.0.0.1:56789'``."""
        peername = self._writer.transport.get_extra_info("peername")
        if peername:
            return f"{peername[0]}:{peername[1]}"
        return "<unknown>"

    # ------------------------------------------------------------------
    # Audio I/O
    # ------------------------------------------------------------------

    async def read_audio(self) -> bytes:
        """Return the raw payload of the next AUDIO (0x10) frame.

        Non-audio frames that arrive in the stream (e.g. DTMF) are silently
        skipped so callers never deal with mixed frame types.

        Returns
        -------
        bytes
            Exactly :data:`FRAME_BYTES` (320) bytes of SLIN signed-16-bit
            little-endian 8 kHz mono PCM.

        Raises
        ------
        AudioSocketHangup
            When a HANGUP frame is received or the TCP connection closes.
        AudioSocketError
            When Asterisk reports an error condition.
        """
        while True:
            msg_type, payload = await _read_frame(self._reader)
            if msg_type == MSG_AUDIO:
                return payload
            # DTMF (0x03) and any unknown extension types are discarded here.
            # If DTMF handling is needed, subclass and override this method.
            log.debug(
                "audiosocket.frame_skipped",
                call_uuid=self._call_uuid,
                msg_type=hex(msg_type),
            )

    async def send_audio(self, pcm: bytes) -> None:
        """Write *pcm* to Asterisk as an AUDIO (0x10) frame.

        Parameters
        ----------
        pcm:
            Exactly :data:`FRAME_BYTES` (320) bytes of SLIN signed-16-bit
            little-endian 8 kHz mono PCM.

        Back-pressure guard
        ~~~~~~~~~~~~~~~~~~~
        If the transport write buffer already holds more than
        ``2 × FRAME_BYTES`` (640 bytes / 40 ms) of unacknowledged data the
        frame is silently discarded.  This prevents unbounded queue growth
        when the caller is producing audio faster than Asterisk drains it —
        which happens during barge-in recovery.

        ``drain()`` is intentionally **not** awaited here.  Awaiting drain
        reintroduces the buffering latency we eliminated with
        ``write_buffer_limits(0)`` and ``TCP_NODELAY``.
        """
        backlog: int = self._writer.transport.get_write_buffer_size()
        if backlog > _MAX_WRITE_BACKLOG:
            log.debug(
                "audiosocket.send_dropped_backlog",
                call_uuid=self._call_uuid,
                backlog_bytes=backlog,
            )
            return

        self._writer.write(_build_frame(MSG_AUDIO, pcm))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Flush any pending writes and close the transport cleanly."""
        try:
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass

        self._writer.close()

        try:
            await self._writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass

        log.info(
            "audiosocket.session_closed",
            call_uuid=self._call_uuid,
            remote_addr=self.remote_addr,
        )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class AudioSocketServer:
    """TCP server that accepts AudioSocket connections from Asterisk.

    Each accepted connection performs the UUID handshake, wraps the socket in
    an :class:`AudioSocketSession`, and invokes *on_call* as an
    :class:`asyncio.Task`.  Multiple concurrent calls are fully supported.

    Parameters
    ----------
    host:
        Bind address, e.g. ``"0.0.0.0"``.
    port:
        TCP port.  Typically 9092 for the main AI agent, 9093 for the
        voicemail/after-hours probe.
    on_call:
        Async callable invoked once per accepted call.  Receives the
        :class:`AudioSocketSession` and should return (or raise
        :exc:`AudioSocketHangup`) when the call ends.  Any exception raised
        inside *on_call* is caught, logged, and does **not** crash the server.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_call: Callable[[AudioSocketSession], Awaitable[None]],
    ) -> None:
        self._host = host
        self._port = port
        self._on_call = on_call
        self._server: asyncio.Server | None = None
        # Strong references prevent the GC from collecting in-flight tasks.
        self._tasks: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------
    # Internal connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Invoked by :func:`asyncio.start_server` for every new TCP connection."""
        # ---- mandatory low-latency socket tuning ----------------------
        raw_sock: socket.socket = writer.transport.get_extra_info("socket")
        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        writer.transport.set_write_buffer_limits(0)

        peername = writer.transport.get_extra_info("peername")
        remote_addr = f"{peername[0]}:{peername[1]}" if peername else "<unknown>"
        log.info("audiosocket.connection_accepted", remote_addr=remote_addr)

        # ---- UUID handshake -------------------------------------------
        call_uuid = ""
        try:
            msg_type, payload = await _read_frame(reader)
        except AudioSocketHangup:
            log.info("audiosocket.hangup_before_uuid", remote_addr=remote_addr)
            writer.close()
            return
        except AudioSocketError as exc:
            log.error(
                "audiosocket.error_during_handshake",
                remote_addr=remote_addr,
                error=str(exc),
            )
            writer.close()
            return

        if msg_type == MSG_UUID:
            call_uuid = payload.decode("utf-8", errors="replace")
            log.info(
                "audiosocket.uuid_received",
                call_uuid=call_uuid,
                remote_addr=remote_addr,
            )
        else:
            # Non-UUID first frame is unusual but not fatal; proceed with an
            # empty uuid so the call handler can still run and log it.
            log.warning(
                "audiosocket.unexpected_first_frame",
                msg_type=hex(msg_type),
                remote_addr=remote_addr,
            )

        session = AudioSocketSession(reader, writer, call_uuid)

        # ---- dispatch to call handler as a tracked task ---------------
        task: asyncio.Task[None] = asyncio.create_task(
            self._run_call(session),
            name=f"audiosocket-{call_uuid or remote_addr}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_call(self, session: AudioSocketSession) -> None:
        """Run *on_call* and guarantee session cleanup on every exit path."""
        try:
            await self._on_call(session)
        except AudioSocketHangup:
            log.info(
                "audiosocket.hangup",
                call_uuid=session.call_uuid,
                remote_addr=session.remote_addr,
            )
        except AudioSocketError as exc:
            log.error(
                "audiosocket.error",
                call_uuid=session.call_uuid,
                remote_addr=session.remote_addr,
                error=str(exc),
            )
        except Exception as exc:
            # Broad catch is intentional: one bad call must not stop the server.
            # BaseException (KeyboardInterrupt, SystemExit, CancelledError) is
            # NOT caught — those must propagate normally.
            log.exception(
                "audiosocket.unhandled_exception",
                call_uuid=session.call_uuid,
                remote_addr=session.remote_addr,
                error=str(exc),
            )
        finally:
            await session.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bind to *host*:*port* and begin accepting AudioSocket connections."""
        self._server = await asyncio.start_server(
            self._handle_connection,
            host=self._host,
            port=self._port,
        )
        bound = ", ".join(
            str(s.getsockname()) for s in self._server.sockets
        )
        log.info(
            "audiosocket.server_started",
            host=self._host,
            port=self._port,
            bound=bound,
        )

    async def stop(self) -> None:
        """Stop accepting new connections and wait for the socket to be released."""
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        log.info(
            "audiosocket.server_stopped",
            host=self._host,
            port=self._port,
        )
