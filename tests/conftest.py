from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest

from proxy import config
from proxy.app import app
from proxy.config import Settings

UpstreamHandler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def proxy_client(monkeypatch):
    """Client talking to the proxy, with the upstream replaced by a mock handler."""

    @asynccontextmanager
    async def build(handler: UpstreamHandler, **overrides) -> AsyncIterator[httpx.AsyncClient]:
        monkeypatch.setattr(config, "settings", Settings(upstream_url="http://upstream.test", **overrides))
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            ) as client:
                yield client
        finally:
            await app.state.client.aclose()

    return build
