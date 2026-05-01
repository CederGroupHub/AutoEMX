#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized logging utilities for the autoemx package.

Library best-practice: the 'autoemx' logger carries only a NullHandler by
default (installed in __init__.py).  Users opt in by adding their own handlers,
e.g. ``logging.basicConfig(level=logging.INFO)``.

For parallel quantification (joblib loky backend), use the helpers below to
route worker log records through a shared ``multiprocessing.Queue`` so that
messages are serialized and streamed to the terminal in real time:

    # Main process — before launching workers
    log_queue, listener = start_parallel_logging()
    try:
        Parallel(n_jobs=N)(delayed(worker)(i, log_queue) for i in items)
    finally:
        stop_parallel_logging(listener)

    # Worker function
    def worker(i, log_queue):
        setup_worker_logging(log_queue)
        logger = get_logger(__name__)
        logger.info("Processing item %d", i)
"""

import logging
import logging.handlers
import multiprocessing
from typing import Any, List, Optional, Tuple


# ── Public API ────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger.  Typically called as ``get_logger(__name__)``."""
    return logging.getLogger(name)


def start_parallel_logging(
    level: int = logging.DEBUG,
) -> Tuple[Optional[multiprocessing.Queue], Optional[logging.handlers.QueueListener]]:
    """
    Create a shared log queue and start a ``QueueListener`` in the main process.

    The listener mirrors log records from worker processes through whatever
    handlers are attached to the ``autoemx`` logger hierarchy (falling back to
    a ``StreamHandler`` if none are configured).

    Parameters
    ----------
    level : int
        Minimum log level forwarded by the listener's handlers.

    Returns
    -------
    queue : multiprocessing.Queue or None
        Pass this to each worker function and call
        ``setup_worker_logging(queue)`` at the top of the worker.
        Returns ``None`` if the queue could not be created.
    listener : QueueListener or None
        Call ``stop_parallel_logging(listener)`` in a ``finally`` block.
        Returns ``None`` if setup failed.
    """
    try:
        # Use a Manager-backed queue so loky workers can pickle and receive it.
        manager = multiprocessing.Manager()
        queue = manager.Queue(-1)
    except Exception:
        return None, None

    handlers = _collect_effective_handlers(logging.getLogger("autoemx"))
    if not handlers:
        handlers = [_default_handler()]

    listener = logging.handlers.QueueListener(
        queue, *handlers, respect_handler_level=True
    )
    # Keep a strong reference so the manager process remains alive until listener.stop().
    listener._autoemx_manager = manager
    listener.start()
    return queue, listener


def stop_parallel_logging(
    listener: Optional[logging.handlers.QueueListener],
) -> None:
    """Stop the ``QueueListener`` returned by :func:`start_parallel_logging`."""
    if listener is not None:
        listener.stop()
        manager = getattr(listener, "_autoemx_manager", None)
        if manager is not None:
            manager.shutdown()


def setup_worker_logging(
    queue: Optional[Any],
    level: int = logging.DEBUG,
) -> None:
    """
    Configure the root logger inside a parallel worker process to route all
    records through *queue*.

    Call at the very top of the worker function, passing the queue returned
    by :func:`start_parallel_logging`.  Has no effect if *queue* is ``None``
    (i.e. parallel logging was not started or queue creation failed).
    """
    if queue is None:
        return
    root = logging.getLogger()
    # Clear any handlers inherited via fork / cloudpickle to avoid duplicates.
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(level)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _collect_effective_handlers(logger: logging.Logger) -> List[logging.Handler]:
    """Walk the logger hierarchy and collect all non-NullHandler handlers."""
    handlers: List[logging.Handler] = []
    current: Optional[logging.Logger] = logger
    while current is not None:
        for h in current.handlers:
            if not isinstance(h, logging.NullHandler):
                handlers.append(h)
        if not getattr(current, "propagate", True):
            break
        current = getattr(current, "parent", None)
    return handlers


def _default_handler() -> logging.StreamHandler:
    """Fallback StreamHandler used when no user handlers are configured."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(name)s | %(levelname)-8s | %(message)s")
    )
    return handler
