"""Simple in-memory job queue for face swap tasks."""

import uuid
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    total_frames: int = 0
    processed_frames: int = 0
    stage: str = ""  # extracting, swapping, merging, done
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "progress": self.progress,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "stage": self.stage,
            "result_path": self.result_path,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class JobStore:
    """In-memory job store with thread safety."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        with self._lock:
            job = Job(id=str(uuid.uuid4()))
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        total_frames: Optional[int] = None,
        processed_frames: Optional[int] = None,
        stage: Optional[str] = None,
        result_path: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if progress is not None:
                job.progress = min(100, max(0, progress))
            if total_frames is not None:
                job.total_frames = total_frames
            if processed_frames is not None:
                job.processed_frames = processed_frames
            if stage is not None:
                job.stage = stage
            if result_path is not None:
                job.result_path = result_path
            if error is not None:
                job.error = error
            job.updated_at = datetime.utcnow()
            return job


# Global job store
job_store = JobStore()
