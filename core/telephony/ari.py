"""
ARI WebSocket client via aiohttp.

ari-py is abandoned since 2019 and is not used here.
All ARI communication is done with raw aiohttp WebSocket + HTTP requests,
giving full control over reconnection, error handling, and async behaviour.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
import structlog

from config.settings import settings

log = structlog.get_logger(__name__)


class ARIConnectionError(Exception):
    """Raised when the ARI WebSocket connection cannot be established or is lost."""


class ARIRequestError(Exception):
    """Raised when an ARI HTTP REST request returns a non-2xx status code."""


class ARIClient:
    """
    Asterisk REST Interface (ARI) client.

    Maintains a single persistent WebSocket for event streaming and uses
    an aiohttp.ClientSession for all REST calls (originate, hangup, answer).

    Usage::

        client = ARIClient()
        await client.connect()
        try:
            async for event in client.events():
                handle(event)
        finally:
            await client.disconnect()
    """

    def __init__(
        self,
        host: str = settings.freepbx_host,
        user: str = settings.freepbx_ari_user,
        secret: str = settings.freepbx_ari_secret,
        app: str = "coldcallai",
    ) -> None:
        self._host = host
        self._user = user
        self._secret = secret
        self._app = app
        self._base_url = f"http://{host}:8088"
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Open an aiohttp ClientSession and connect the ARI event WebSocket.

        Subscribes to events for ``self._app`` only (subscribeAll=false).
        Raises :exc:`ARIConnectionError` if the WebSocket handshake fails.
        """
        ws_url = (
            f"{self._base_url}/ari/events"
            f"?app={self._app}"
            f"&api_key={self._user}:{self._secret}"
            f"&subscribeAll=false"
        )
        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(ws_url)
        except Exception as exc:
            if self._session and not self._session.closed:
                await self._session.close()
            raise ARIConnectionError(
                f"Failed to connect to ARI at {self._base_url}: {exc}"
            ) from exc

        log.info(
            "ari.connected",
            host=self._host,
            app=self._app,
        )

    async def disconnect(self) -> None:
        """Close the WebSocket and the underlying aiohttp session cleanly."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

        log.info(
            "ari.disconnected",
            host=self._host,
            app=self._app,
        )

    # ------------------------------------------------------------------
    # Event stream
    # ------------------------------------------------------------------

    async def events(self) -> AsyncGenerator[dict[str, Any], None]:
        """
        Async generator that yields parsed JSON event dicts from ARI.

        Exits cleanly when the WebSocket transitions to CLOSING or CLOSED,
        or when the enclosing task is cancelled.  Raises
        :exc:`ARIConnectionError` on an ERROR frame (unexpected
        protocol-level error reported by Asterisk).
        """
        if self._ws is None:
            raise ARIConnectionError("Not connected — call connect() first.")

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    event: dict[str, Any] = msg.json()
                    log.debug(
                        "ari.event_received",
                        event_type=event.get("type"),
                        channel_id=(
                            event.get("channel", {}).get("id")
                            if isinstance(event.get("channel"), dict)
                            else None
                        ),
                    )
                    yield event

                elif msg.type in (
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    log.info(
                        "ari.ws_closed",
                        host=self._host,
                        app=self._app,
                    )
                    break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    raise ARIConnectionError(
                        f"ARI WebSocket error frame: {self._ws.exception()}"
                    )

        except asyncio.CancelledError:
            log.info(
                "ari.events_cancelled",
                host=self._host,
                app=self._app,
            )
            raise

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    def _auth(self) -> aiohttp.BasicAuth:
        return aiohttp.BasicAuth(self._user, self._secret)

    async def originate(
        self,
        endpoint: str,
        caller_id: str,
        channel_id: str,
        context: str = "coldcallai-outbound",
        extension: str = "s",
        priority: int = 1,
    ) -> dict[str, Any]:
        """
        Originate an outbound channel via ARI POST /ari/channels.

        Parameters
        ----------
        endpoint:
            Asterisk dial string, e.g. ``"PJSIP/+15551234567@trunk"``.
        caller_id:
            Caller-ID presented to the called party.
        channel_id:
            Caller-supplied unique ID used to correlate subsequent ARI
            events (StasisStart, ChannelHangupRequest, etc.).
        context / extension / priority:
            Dialplan co-ordinates the channel enters after answer.

        Returns the Asterisk channel object dict.
        Raises :exc:`ARIRequestError` on any non-2xx HTTP response.
        """
        if self._session is None:
            raise ARIConnectionError("Not connected — call connect() first.")

        url = f"{self._base_url}/ari/channels"
        payload: dict[str, Any] = {
            "endpoint": endpoint,
            "callerId": caller_id,
            "channelId": channel_id,
            "context": context,
            "extension": extension,
            "priority": priority,
            "app": self._app,
        }

        log.info(
            "ari.originate",
            endpoint=endpoint,
            channel_id=channel_id,
            caller_id=caller_id,
        )

        async with self._session.post(url, json=payload, auth=self._auth()) as resp:
            if resp.status not in range(200, 300):
                body = await resp.text()
                raise ARIRequestError(
                    f"ARI originate failed [{resp.status}] endpoint={endpoint}: {body}"
                )
            return await resp.json()  # type: ignore[no-any-return]

    async def hangup(self, channel_id: str) -> None:
        """
        Delete (hang up) a channel via ARI DELETE /ari/channels/{channel_id}.

        A 404 response is silently ignored — the channel is already gone.
        Raises :exc:`ARIRequestError` on any other non-2xx status.
        """
        if self._session is None:
            raise ARIConnectionError("Not connected — call connect() first.")

        url = f"{self._base_url}/ari/channels/{channel_id}"

        log.info("ari.hangup", channel_id=channel_id)

        async with self._session.delete(url, auth=self._auth()) as resp:
            if resp.status == 404:
                log.debug("ari.hangup_already_gone", channel_id=channel_id)
                return
            if resp.status not in range(200, 300):
                body = await resp.text()
                raise ARIRequestError(
                    f"ARI hangup failed [{resp.status}] channel_id={channel_id}: {body}"
                )

    async def answer(self, channel_id: str) -> None:
        """
        Answer a ringing channel via ARI POST /ari/channels/{channel_id}/answer.

        Raises :exc:`ARIRequestError` on non-2xx.
        """
        if self._session is None:
            raise ARIConnectionError("Not connected — call connect() first.")

        url = f"{self._base_url}/ari/channels/{channel_id}/answer"

        async with self._session.post(url, auth=self._auth()) as resp:
            if resp.status not in range(200, 300):
                body = await resp.text()
                raise ARIRequestError(
                    f"ARI answer failed [{resp.status}] channel_id={channel_id}: {body}"
                )


# ---------------------------------------------------------------------------
# Module-level singleton — lazy, no I/O at import time
# ---------------------------------------------------------------------------

ari_client = ARIClient()
