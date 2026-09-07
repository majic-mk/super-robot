"""Premeasurement evidence journal, not an online or GPU qualification gate.

The immutable measurement *plan* is known before CUDA measurements exist.
Keep their digest null here, then bind a measured child manifest later. The
online event log deliberately continues requiring actual measurement evidence.
"""
from pathlib import Path
from threading import RLock

from .v8_schema10_cost_collection import STAGES
from .v8_schema10_event_log import OnlineEventLog, atomic_json, read_events
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest


class StageEvidenceJournal:
    # Reuse only the hash-chain append primitive, not online binding/resume rules.
    append = OnlineEventLog.append

    def __init__(self, path, *, binding, jobs, resume=False):
        required = {"code_commit", "patch_sha256", "model_signature", "model_revision",
                    "tokenizer_hash", "config_sha256", "initial_state_sha256",
                    "measurement_plan_sha256", "gpu_uuid"}
        if (not required <= binding.keys() or any(not binding[k] for k in required)
                or binding.get("runtime_measurement_sha256") is not None
                or binding.get("evidence_scope") != "native_staged_prequalification"):
            raise ValueError("staged evidence requires a plan binding, not an invented cost digest")
        self.jobs = list(jobs)
        if (not self.jobs or any(set(j) != {"job_id", "stage", "input_sha256"} for j in self.jobs)
                or any(j["stage"] not in STAGES or not j["input_sha256"] or not j["job_id"] for j in self.jobs)
                or len({j["job_id"] for j in self.jobs}) != len(self.jobs)
                or [STAGES.index(j["stage"]) for j in self.jobs] != sorted(STAGES.index(j["stage"]) for j in self.jobs)):
            raise ValueError("staged jobs must be unique, immutable and ordered")
        self.binding = {**binding, "ordered_jobs_sha256": digest_json(self.jobs)}
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.active = None
        self.failed = False
        self.completed = []
        if self.path.exists():
            if not resume:
                raise FileExistsError("never overwrite stage evidence")
            self.rows = read_events(self.path, binding=self.binding)
            self._validate_success_prefix()
        else:
            self.path.touch(exist_ok=False)
            self.rows = []

    def _validate_success_prefix(self):
        if len(self.rows) % 2:
            raise ValueError("incomplete stage job; preserve evidence and use a new output directory")
        for index in range(0, len(self.rows), 2):
            if index // 2 >= len(self.jobs):
                raise ValueError("unexpected completed stage job")
            job = self.jobs[index // 2]
            start, end = self.rows[index:index + 2]
            if (start["kind"] != "stage_job_started" or end["kind"] != "stage_job_completed"
                    or start["request_id"] != job["job_id"] or end["request_id"] != job["job_id"]
                    or start["payload"] != job):
                raise ValueError("only an ordered, failure-free successful prefix can resume")
            descriptor = end["payload"]
            if descriptor.get("job_id") != job["job_id"] or descriptor.get("stage") != job["stage"]:
                raise ValueError("completed artifact belongs to another job/stage")
            path = self.path.parent / descriptor["relative_path"]
            if (path.resolve().parent != self.path.parent.resolve()
                    or not path.is_file() or file_digest(path) != descriptor["sha256"]):
                raise ValueError("completed stage output missing, outside journal directory, or corrupted")
            self.completed.append(descriptor)

    def begin(self, job_id):
        with self.lock:
            if self.active is not None or len(self.completed) >= len(self.jobs):
                raise RuntimeError("stage job is active or all jobs are completed")
            job = self.jobs[len(self.completed)]
            if job_id != job["job_id"]:
                raise ValueError("cannot skip a prerequisite stage/job")
            self.append("stage_job_started", job_id, job)
            self.active = job

    def complete(self, result, *, validate):
        """Require raw-result validation; `passed=True` alone is never sufficient.

        Validators come from the concrete native stage dispatcher, not JSON
        input. Successful resume additionally needs backend state reconstruction;
        this journal certifies file integrity only, never live CUDA restoration.
        """
        with self.lock:
            if self.active is None or self.failed:
                raise RuntimeError("completion without an active job")
            if not callable(validate):
                raise ValueError("raw result validator required")
            if validate(result) is not True:
                raise ValueError("raw stage result did not pass its validator")
            name = f"stage-result-{len(self.completed):05d}.json"
            path = self.path.parent / name
            if path.exists():
                raise FileExistsError("never overwrite stage output")
            atomic_json(path, result)
            descriptor = {"job_id": self.active["job_id"], "stage": self.active["stage"],
                          "relative_path": name, "sha256": file_digest(path)}
            self.append("stage_job_completed", self.active["job_id"], descriptor)
            self.completed.append(descriptor)
            self.active = None
            return descriptor

    def fail(self, error):
        with self.lock:
            if self.active is None:
                raise RuntimeError("failure without an active job")
            self.append("stage_job_failed", self.active["job_id"],
                        {"error_type": type(error).__name__, "message": str(error)})
            self.failed = True
            # Keep the failed job active: this session cannot advance past it.
