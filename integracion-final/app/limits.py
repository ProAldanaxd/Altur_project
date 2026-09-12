"""Acota el cuerpo antes de que FastAPI cargue el JSON, incluso sin Content-Length."""
from starlette.responses import JSONResponse

from app.audio import MAX_BASE64_CHARS

MAX_REQUEST_BYTES = MAX_BASE64_CHARS + 65536


class RequestSizeLimit:
    def __init__(self, app, max_bytes=MAX_REQUEST_BYTES):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] not in {"/detect", "/conversation/analyze", "/conversation/semantic", "/voice/alert"} or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        limit = min(self.max_bytes, 512 * 1024) if scope["path"] == "/conversation/semantic" else self.max_bytes
        if scope["path"] == "/voice/alert":
            limit = min(limit, 1024)
        headers = dict(scope.get("headers", []))
        if b"content-length" in headers:
            try:
                declared = int(headers[b"content-length"])
                if declared < 0:
                    raise ValueError()
            except ValueError:
                return await JSONResponse({"detail": "Content-Length inválido"}, status_code=400)(scope, receive, send)
            if declared > limit:
                return await JSONResponse({"detail": "Solicitud demasiado grande"}, status_code=413)(scope, receive, send)
        messages, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > limit:
                return await JSONResponse({"detail": "Solicitud demasiado grande"}, status_code=413)(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break
        iterator = iter(messages)

        async def replay():
            try:
                return next(iterator)
            except StopIteration:
                return await receive()

        await self.app(scope, replay, send)
