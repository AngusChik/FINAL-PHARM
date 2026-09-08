"""Local catalogue learning that runs alongside the website process."""

import logging
import threading

from django.conf import settings
from django.db import close_old_connections, connections

from .prescription_drug_learning import learn_prescription_drugs


logger = logging.getLogger(__name__)
_BATCH_SIZE = 100
_BACKLOG_DELAY_SECONDS = 1
_start_lock = threading.Lock()
_worker_thread = None


def _interval_seconds():
    try:
        return max(1, int(getattr(
            settings, 'PRESCRIPTION_DRUG_LEARNING_INTERVAL_SECONDS', 60,
        )))
    except (TypeError, ValueError):
        return 60


def _run_worker(stop_event):
    """Retry durable source rows after failures and release idle connections."""
    while not stop_event.is_set():
        delay = _interval_seconds()
        try:
            close_old_connections()
            result = learn_prescription_drugs(batch_size=_BATCH_SIZE)
            if result.get('processed', 0) >= _BATCH_SIZE:
                delay = _BACKLOG_DELAY_SECONDS
        except Exception as exc:
            # Database exceptions may include source text. Keep patient and
            # drug details out of background-process logs and tracebacks.
            logger.warning(
                'Prescription drug learning pass failed (%s); retrying automatically.',
                type(exc).__name__,
            )
        finally:
            try:
                connections.close_all()
            except Exception as exc:
                logger.warning(
                    'Prescription drug learning connection cleanup failed (%s).',
                    type(exc).__name__,
                )
        if stop_event.wait(delay):
            break


def start_prescription_drug_worker():
    """Start once per web process; command-line setup never calls this hook."""
    global _worker_thread

    if not getattr(settings, 'PRESCRIPTION_DRUG_LEARNING_ENABLED', True):
        return False
    with _start_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return False
        worker = threading.Thread(
            target=_run_worker,
            args=(threading.Event(),),
            name='prescription-drug-learning',
            daemon=True,
        )
        try:
            worker.start()
        except Exception as exc:
            logger.warning(
                'Prescription drug learning worker could not start (%s).',
                type(exc).__name__,
            )
            return False
        _worker_thread = worker
        return True
