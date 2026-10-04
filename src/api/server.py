"""
src/api/server.py
=================
Application factory and server entry point for AVA LDM Studio API.
"""

from __future__ import annotations

import logging
from typing import Optional

from aiohttp import web

from .routes import LDMRoutes
from .service import LDMService

logger = logging.getLogger(__name__)


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Enable Cross-Origin Resource Sharing for Android emulators and web clients."""
    if request.method == "OPTIONS":
        response = web.Response(status=204)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as ex:
            response = ex

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, user_id"
    return response


def create_app(service: Optional[LDMService] = None) -> web.Application:
    """
    Create and configure the aiohttp Application.
    """
    app = web.Application(middlewares=[cors_middleware])
    routes = LDMRoutes(service=service)
    routes.setup_routes(app)
    return app


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    service: Optional[LDMService] = None,
) -> None:
    """Run the local aiohttp server."""
    app = create_app(service=service)
    logger.info(f"Starting AVA LDM Studio API on http://{host}:{port}")
    web.run_app(app, host=host, port=port)
