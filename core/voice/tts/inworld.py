"""
InWorld TTS provider — streaming WebSocket implementation.

INWORLD API — VERIFY BEFORE USE:
  Before this file goes live, confirm with InWorld docs:
  1. WebSocket endpoint URL      → set INWORLD_TTS_ENDPOINT in .env
  2. Auth header                 → currently "Authorization: Bearer {api_key}"
                                   may be "X-Api-Key" or a custom header
  3. Synthesis request format    → see synthesize() → send_str() call below
                                   verify field names: "type", "voice_id", "text"
  4. Audio chunk format received → currently expects raw binary (WSMsgType.BINARY)
                                   may be base64-encoded JSON instead
  5. Completion signal           → currently {"type": "synthesis_complete"}
                                   verify the exact field name InWorld sends
  6. Audio encoding              → InWorld Realtime TTS-2 may output 16kHz linear PCM
                                   If so, resample 16kHz → 8kHz before sending to FreePBX
                                   (see core/telephony/ulaw.py — resample_16k_to_8k)
"""

import asyncio
import json
from typing import AsyncGenerator, Optional

import aiohttp
import structlog

from .base import TTSProvider

logger = structlog.get_logger()


class TTSConnectionError(Exception):
    pass


class InWorldTTSProvider(TTSProvider):
    """
    Streams text to InWorld Realtime TTS-2 over a persistent WebSocket.
    Yields 8kHz/16-bit/mono PCM audio chunks as they arrive (~200ms each).

    Barge-in: call stop() to cancel ongoing synthesis and reset the connection.
    The WS is reconnected on stop() to flush any buffered audio InWorld may
    have queued for the cancelled utterance (~20ms overhead, well within 100ms).
    """

    def __init__(self, api_key: str, endpoint: str) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        # Set by stop() to break the synthesize loop immediately.
        # Cleared by stop() after reconnecting so the next synthesize() is clean.
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        # Reuse session across reconnects; only create if missing or closed.
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Basic {self._api_key}"}
            )
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = await self._session.ws_connect(self._endpoint)
        logger.info("inworld_tts_connected", endpoint=self._endpoint)

    async def disconnect(self) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("inworld_tts_disconnected")

    # ------------------------------------------------------------------
    # Barge-in stop
    # ------------------------------------------------------------------

    async def stop(self) -> None:
        """
        Cancel ongoing synthesis and reset to clean state for the next call.

        1. Sets _stop_event  → synthesize loop breaks on next iteration
        2. Closes the WS    → flushes any buffered InWorld audio for this utterance
        3. Reconnects       → fresh WS, no stale messages
        4. Clears _stop_event → next synthesize() call is unblocked
        """
        self._stop_event.set()
        if self._ws and not self._ws.closed:
            await self._ws.close()
        await self.connect()
        self._stop_event.clear()
        logger.info("inworld_tts_stopped_and_reset")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    async def synthesize(
        self, text: str, voice_id: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Send text to InWorld, yield PCM audio chunks as they arrive.
        Stops early if stop() is called (barge-in).
        """
        if not text.strip():
            raise ValueError("text must not be empty")

        if self._ws is None:
            raise TTSConnectionError("Not connected — call connect() first")
        # InWorld closes the WS after each utterance; reconnect lazily when closed.
        # Do NOT always-reconnect: rapid reconnects trigger InWorld rate limiting.
        if self._ws.closed:
            await self.connect()

        req = json.dumps({"text": text, "voiceId": voice_id, "modelId": "inworld-tts-2"})
        try:
            await self._ws.send_str(req)
        except (ConnectionResetError, aiohttp.ClientConnectionError):
            # WS closing but not yet marked closed — reconnect and retry once.
            await asyncio.sleep(0.5)  # brief pause to avoid InWorld rate limit
            await self.connect()
            await self._ws.send_str(req)

        async for msg in self._ws:
            # Check barge-in signal before processing the message
            if self._stop_event.is_set():
                break

            if msg.type == aiohttp.WSMsgType.BINARY:
                logger.debug("inworld_tts_raw_binary", size=len(msg.data))
                yield msg.data

            elif msg.type == aiohttp.WSMsgType.TEXT:
                logger.debug("inworld_tts_raw_text", raw=msg.data[:500])
                payload = self._parse_text(msg.data)
                if payload and payload.get("type") == "synthesis_complete":
                    break  # utterance complete — clean exit

            elif msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                logger.debug("inworld_tts_ws_closed",
                             code=getattr(msg, 'data', None),
                             extra=getattr(msg, 'extra', None))
                break  # connection closed cleanly — stop yielding

            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise TTSConnectionError(
                    f"WebSocket error during synthesis: {self._ws.exception()}"
                )

    def _parse_text(self, raw: str) -> Optional[dict]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("inworld_tts_unparseable_message", raw=raw[:200])
            return None
