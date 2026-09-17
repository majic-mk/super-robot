"""Optional local-disk evidence journal; fsync remains on the request path."""
import os
import shutil
import time
from pathlib import Path

from .v8_schema10_event_log import read_events
from .v8_schema10_storage import file_digest


def journal_path(output, directory=None):
    if directory is None:
        return Path(output) / 'events.jsonl'
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError('external journal directory must be absolute')
    path.mkdir(parents=True, exist_ok=False)
    return path / 'events.jsonl'


def archive_journal(source, destination, *, binding):
    start = time.perf_counter_ns()
    source, destination = Path(source), Path(destination)
    rows = read_events(source, binding=binding)
    digest = file_digest(source)
    if source.resolve() != destination.resolve():
        if destination.exists():
            raise FileExistsError('never overwrite archived evidence')
        temporary = destination.with_name(destination.name + '.archiving')
        with source.open('rb') as incoming, temporary.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if file_digest(temporary) != digest or file_digest(source) != digest:
            raise ValueError('journal changed during archive; preserve both files')
        os.replace(temporary, destination)
    return dict(source_path=str(source), archived_path=str(destination),
                sha256=digest, event_count=len(rows), source_retained=True,
                archive_wall_ms=(time.perf_counter_ns() - start) / 1e6,
                request_fsync_enabled=True)
