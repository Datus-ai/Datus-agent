"""Keep a service alive through streaming on every supported FastAPI version."""

from starlette.responses import StreamingResponse

from datus.utils.artifact_output import artifact_request_scope, current_artifact_slug_factory


class ServiceStreamingResponse(StreamingResponse):
    def __init__(self, content, *, service, **kwargs):
        super().__init__(content, **kwargs)
        self._service = service
        # Yield dependencies can close before response iteration. Preserve the
        # allocator while the handler still runs in the authenticated context.
        self._artifact_slug_factory = current_artifact_slug_factory(getattr(service, "agent_config", None))
        # Older FastAPI versions exit yield dependencies before streaming.
        # Retain while the handler still owns its original request borrow.
        acquire = getattr(service, "acquire_request", None)
        self._pinned = acquire(allow_closing=True) if callable(acquire) else False

    async def __call__(self, scope, receive, send):
        try:
            with artifact_request_scope(factory=self._artifact_slug_factory):
                await super().__call__(scope, receive, send)
        finally:
            if self._pinned:
                self._pinned = False
                self._service.release_request()
