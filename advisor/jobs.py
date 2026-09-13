"""Background training jobs for the web admin area.

A genetic-algorithm run takes minutes, far longer than a web request should
block.  `JobManager` runs `train_rule` in a daemon thread and exposes a
thread-safe snapshot (status, progress log, result or error) that the admin
page polls.  One job runs at a time: the GA is CPU-bound, so concurrent runs
would only slow each other down, and a single slot keeps the UI simple.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field

from .pipeline import TrainRequest, train_rule


@dataclass
class Job:
    request: dict
    status: str = "queued"           # queued | running | done | failed
    log: list[str] = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    generation: int = 0              # last generation reported
    generations: int = 0             # total requested

    def snapshot(self) -> dict:
        elapsed = None
        if self.started_at:
            elapsed = (self.finished_at or time.time()) - self.started_at
        return {
            "request": self.request,
            "status": self.status,
            "log": list(self.log),
            "result": self.result,
            "error": self.error,
            "elapsed_s": elapsed,
            "generation": self.generation,
            "generations": self.generations,
            "progress_pct": (100 * self.generation // self.generations) if self.generations else 0,
        }


class JobManager:
    """Owns at most one active job plus a short history of finished ones."""

    def __init__(self, max_history: int = 10):
        self._lock = threading.Lock()
        self._current: Job | None = None
        self._history: list[Job] = []
        self._max_history = max_history

    # ---------------------------------------------------------------- API
    def is_busy(self) -> bool:
        with self._lock:
            return self._current is not None and self._current.status in ("queued", "running")

    def current(self) -> dict | None:
        with self._lock:
            return self._current.snapshot() if self._current else None

    def history(self) -> list[dict]:
        with self._lock:
            return [j.snapshot() for j in self._history]

    def start(self, req: TrainRequest) -> Job:
        """Validate and launch a job; raises RuntimeError if one is running."""
        req.validate()
        with self._lock:
            if self._current and self._current.status in ("queued", "running"):
                raise RuntimeError("A training job is already running")
            if self._current:
                self._history.insert(0, self._current)
                del self._history[self._max_history:]
            job = Job(request=req.to_dict(), generations=req.generations)
            self._current = job

        thread = threading.Thread(target=self._run, args=(job, req), daemon=True)
        thread.start()
        return job

    # ------------------------------------------------------------ worker
    def _run(self, job: Job, req: TrainRequest) -> None:
        def progress(msg: str) -> None:
            with self._lock:
                job.log.append(msg)
                if msg.startswith("gen "):
                    try:
                        job.generation = int(msg.split()[1]) + 1
                    except (IndexError, ValueError):
                        pass

        with self._lock:
            job.status = "running"
            job.started_at = time.time()
        try:
            outcome = train_rule(req, progress=progress)
            with self._lock:
                job.result = outcome.to_dict()
                job.status = "done"
                job.generation = job.generations
        except Exception as exc:  # surfaced to the admin page, never crashes the server
            with self._lock:
                job.error = f"{type(exc).__name__}: {exc}"
                job.log.append("Traceback:\n" + traceback.format_exc())
                job.status = "failed"
        finally:
            with self._lock:
                job.finished_at = time.time()
