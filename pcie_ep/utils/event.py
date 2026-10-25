"""CUDA event overlap and allocator stream ownership helpers."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch


class EventOverlap:
    """A CUDA stream dependency, with the DeepEP overlap/context interface.

    ``wait`` and ``current_stream_wait`` enqueue a device-side wait.  Neither
    synchronizes the calling CPU thread.  ``event`` is a torch CUDA Event.
    """

    def __init__(
        self, event: Any = None, extra_tensors: Any = None, on_wait: Optional[Callable[[Any], None]] = None
    ) -> None:
        """Retain the completion event and objects needed by queued GPU work.

        Arguments:
            event: native torch CUDA dependency event.
            extra_tensors: nested tensors and owner references kept alive.
            on_wait: buffer callback registering the current consumer stream;
                this protects manual output-pool reuse as well as allocation.
        """
        self.event = event
        self.extra_tensors = extra_tensors
        self._on_wait = on_wait
        self.hook_after_wait: Optional[Callable[[], Any]] = None
        self._release_handle_by_call = False

    def current_stream_wait(self, release_handle: bool = False) -> Any:
        """Enqueue a GPU wait, register the consumer, and run one optional hook.

        The calling CPU never waits for completion.  Releasing this wrapper's
        references does not release a handle's independent diagnostic event.
        """
        if self.event is None:
            raise RuntimeError("The event has already been released")
        stream = torch.cuda.current_stream()
        stream.wait_event(self.event)
        # This keeps allocator ownership correct even when the caller waits on
        # a stream other than the one which submitted dispatch.
        _record_tensors(self.extra_tensors, stream)
        if self._on_wait is not None:
            self._on_wait(stream)
        result = None
        if self.hook_after_wait is not None:
            hook, self.hook_after_wait = self.hook_after_wait, None
            result = hook()
        if release_handle:
            self.event = None
            self.extra_tensors = None
            self._on_wait = None
        return result

    def wait(self) -> Any:
        """Alias for current_stream_wait; return the one-shot hook result."""
        return self.current_stream_wait()

    def register_hook_after_wait(self, hook_after_wait: Callable[[], Any]) -> None:
        """Register an epilogue result hook to run once after the next wait."""
        if self.hook_after_wait is not None:
            raise RuntimeError("An epilogue hook is already registered")
        self.hook_after_wait = hook_after_wait

    def __call__(self, release_handle: bool = False) -> "EventOverlap":
        """Configure release behavior for the following context-manager exit."""
        self._release_handle_by_call = release_handle
        return self

    def __enter__(self) -> "EventOverlap":
        """Allow independent compute inside the scope; wait when it exits."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if self.event is not None:
                self.current_stream_wait(self._release_handle_by_call)
        finally:
            self._release_handle_by_call = False


def _record_tensors(value: Any, stream: Any) -> None:
    """Record nested CUDA tensors with the allocator, ignoring owner objects.

    This does not prevent a runtime from overwriting its still-allocated output
    pool.  EventOverlap's on_wait callback handles that separate dependency.
    """
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for child in value.values():
            _record_tensors(child, stream)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _record_tensors(child, stream)
