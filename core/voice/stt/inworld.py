"""
InWorld STT provider — streaming WebSocket implementation.

INWORLD API — VERIFY BEFORE USE:
  Before this file goes live, confirm with InWorld docs:
  1. WebSocket endpoint URL  → set INWORLD_STT_ENDPOINT in .env
  2. Auth header             → currently "Authorization: Bearer {api_key}"
                               may be "X-Api-Key" or a custom header
  3. Audio send format       → currently raw bytes (PCM µ-law 8kHz)
                               may need base64 or a JSON wrapper
  4. Message format received → see _parse_message() below
                               field names must match InWorld's actual schema
  5. End-of-stream signal    → currently {"type": "end_stream"}
                               InWorld may use a different signal or none
"""

import asyncio
import json
from typing import AsyncGenerator, Optional

import aiohttp
import structlog

from .base import FinalTranscript, PartialTranscript, STTEvent, STTProvider

logger = structlog.get_logger()

_MAX_RECONNECT_ATTEMPTS = 3
_RECONNECT_DELAY_SECS = 0.5


class STTConnectionError(Exception):
    pass


class InWorldSTTProvider(STTProvider):
    """
    Streams 8kHz/16-bit/mono PCM to InWorld STT over a WebSocket.
    Emits PartialTranscript every ~50ms and FinalTranscript on end-of-utterance.
    Reconnects silently up to 3 times on connection drop.
    """

    def __init__(self, api_key: str, endpoint: str) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Basic {self._api_key}"}
        )
        self._ws = await self._session.ws_connect(self._endpoint)
        logger.info("inworld_stt_connected", endpoint=self._endpoint)

    async def disconnect(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("inworld_stt_disconnected")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream_audio(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[STTEvent, None]:
        """
        Send audio chunks and yield transcript events.
        InWorld closes the WS after each transcription request, so we
        always reconnect at the start of each turn.
        """
        # InWorld is one-connection-per-request; reconnect fresh every turn.
        await self.connect()

        last_error: Optional[Exception] = None
        for attempt in range(_MAX_RECONNECT_ATTEMPTS):
            try:
                async for event in self._stream_once(audio_chunks):
                    yield event
                return  # clean finish
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                last_error = exc
                logger.warning(
                    "inworld_stt_reconnecting",
                    attempt=attempt + 1,
                    max=_MAX_RECONNECT_ATTEMPTS,
                    error=str(exc),
                )
                if attempt < _MAX_RECONNECT_ATTEMPTS - 1:
                    await asyncio.sleep(_RECONNECT_DELAY_SECS)
                    await self.connect()

        raise STTConnectionError(
            f"InWorld STT failed after {_MAX_RECONNECT_ATTEMPTS} attempts"
        ) from last_error

    async def _stream_once(
        self, audio_chunks: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[STTEvent, None]:
        """Single-connection streaming attempt. Raises on WS error."""
        if self._ws is None or self._ws.closed:
            raise aiohttp.ClientError("WebSocket not connected")

        send_task = asyncio.create_task(self._send_audio(audio_chunks))
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    logger.debug("inworld_stt_raw_text", raw=msg.data[:500])
                    event = self._parse_message(msg.data)
                    if event is not None:
                        yield event
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    logger.debug("inworld_stt_raw_binary", size=len(msg.data))
                    # InWorld may send binary — ignore unless their docs say otherwise
                    pass
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise aiohttp.ClientError(
                        f"WebSocket error: {self._ws.exception()}"
                    )
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    logger.debug("inworld_stt_ws_closed",
                                 code=getattr(msg, 'data', None),
                                 extra=getattr(msg, 'extra', None))
                    break
        finally:
            send_task.cancel()
            try:
                await send_task
            except asyncio.CancelledError:
                pass

    async def _send_audio(self, audio_chunks: AsyncGenerator[bytes, None]) -> None:
        """
        Send streaming config first, then pump audio bytes.
        Signal end-of-stream by closing the WS (InWorld reads EOF on close).
        """
        if not (self._ws and not self._ws.closed):
            return

        # Initial config message required by InWorld before audio bytes.
        await self._ws.send_str(json.dumps({
            "streamingConfig": {
                "config": {
                    "encoding": "LINEAR16",
                    "sampleRateHertz": 8000,
                    "languageCode": "en-US",
                }
            }
        }))

        async for chunk in audio_chunks:
            if self._ws and not self._ws.closed:
                await self._ws.send_bytes(chunk)

        # Close WS from our side to signal end-of-stream (no special JSON needed).
        if self._ws and not self._ws.closed:
            await self._ws.close()

    # ------------------------------------------------------------------
    # Message parsing
    # ------------------------------------------------------------------

    def _parse_message(self, raw: str) -> Optional[STTEvent]:
        """
        Parse one JSON message from InWorld STT.

        InWorld uses Google Cloud STT format:
          {"results": [{"alternatives": [{"transcript": "...", "confidence": 0.9}], "isFinal": true}]}

        Also handles a simpler flat format as fallback:
          {"type": "partial"|"final", "transcript": "..."}
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("inworld_stt_unparseable_message", raw=raw[:200])
            return None

        # Google Cloud STT / InWorld format
        results = msg.get("results")
        if results:
            for result in results:
                alts = result.get("alternatives", [])
                if not alts:
                    continue
                transcript = alts[0].get("transcript", "").strip()
                if not transcript:
                    continue
                is_final = result.get("isFinal", False)
                confidence = float(alts[0].get("confidence", 1.0))
                if is_final:
                    return FinalTranscript(text=transcript, confidence=confidence)
                return PartialTranscript(text=transcript)

        # Flat fallback format
        msg_type = msg.get("type")
        transcript = msg.get("transcript", "").strip()
        if transcript:
            if msg_type == "final":
                return FinalTranscript(text=transcript, confidence=float(msg.get("confidence", 1.0)))
            if msg_type == "partial":
                return PartialTranscript(text=transcript)

        return None
