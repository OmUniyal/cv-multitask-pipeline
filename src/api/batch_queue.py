import asyncio
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class InferenceRequest:
    """
    A single inference request waiting in the queue.
    future: asyncio.Future the HTTP handler is awaiting.
    image_tensor: preprocessed image tensor ready for model.
    request_id: unique ID for logging.
    enqueue_time: when the request entered the queue.
    """
    future: asyncio.Future
    image_tensor: object      # torch.Tensor
    request_id: str
    enqueue_time: float = field(default_factory=time.time)


class BatchQueue:
    """
    Async batch accumulator for inference requests.

    How it works:
    1. HTTP handler preprocesses image, creates a Future, adds to queue
    2. Handler awaits the Future (suspends, doesn't block the server)
    3. Background worker wakes every max_wait_ms OR when batch is full
    4. Worker collects pending requests, runs one batched forward pass
    5. Worker sets results on each Future — all handlers wake up simultaneously

    This means one model forward pass can serve multiple users at once.
    """

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._queue: List[InferenceRequest] = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()

    async def enqueue(
        self,
        image_tensor,
        request_id: str,
    ):
        """
        Add an image to the queue and wait for its result.
        The HTTP handler calls this and awaits — it suspends here
        until the batch worker processes its request.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = InferenceRequest(
            future=future,
            image_tensor=image_tensor,
            request_id=request_id,
        )

        async with self._lock:
            self._queue.append(request)
            # signal the worker that there's work to do
            self._batch_event.set()

        # suspend here until worker sets the result
        return await future

    async def get_batch(self) -> List[InferenceRequest]:
        """
        Wait until we have requests, then return a batch.
        Wakes up when:
        - batch is full (max_batch_size reached), OR
        - max_wait_ms timeout expires
        Whichever comes first.
        """
        # wait for at least one request
        await self._batch_event.wait()

        # wait up to max_wait_ms for more requests to accumulate
        await asyncio.sleep(self.max_wait_ms / 1000.0)

        async with self._lock:
            # take up to max_batch_size requests
            batch = self._queue[:self.max_batch_size]
            self._queue = self._queue[self.max_batch_size:]

            # reset event if queue is now empty
            if not self._queue:
                self._batch_event.clear()

        return batch

    def set_results(
        self,
        batch: List[InferenceRequest],
        results: List,
    ):
        """
        Set results on each request's Future.
        This wakes up all the HTTP handlers that were awaiting.
        """
        for request, result in zip(batch, results):
            if not request.future.done():
                request.future.set_result(result)

    def set_error(
        self,
        batch: List[InferenceRequest],
        error: Exception,
    ):
        """Set an exception on all requests in a failed batch."""
        for request in batch:
            if not request.future.done():
                request.future.set_exception(error)

    @property
    def queue_size(self) -> int:
        return len(self._queue)