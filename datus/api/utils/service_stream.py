"""Keep a service alive through streaming on every supported FastAPI version."""

from starlette.responses import StreamingResponse


class ServiceStreamingResponse(StreamingResponse):
    def __init__(self, content, *, service, **kwargs):
        super().__init__(content, **kwargs)
        self._service = service
        # Older FastAPI versions exit yield dependencies before streaming.
        # Retain while the handler still owns its original request borrow.
        acquire = getattr(service, "acquire_request", None)
        self._pinned = acquire(allow_closing=True) if callable(acquire) else False

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self._pinned:
                self._pinned = False
                self._service.release_request()
