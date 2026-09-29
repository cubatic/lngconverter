import time
from pathlib import Path

import requests

from .config import Settings
from .pipeline import PipelineError, extract_audio
from .source import ResolvedSource, resolve_source


class IngestWorker:
    def __init__(self, server_url: str, ingest_key: str, settings: Settings, work_dir: Path):
        self.server_url = server_url.rstrip("/")
        self.settings = settings
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["x-lgn-ingest-key"] = ingest_key
        # yt-dlp resolution is comparatively expensive. A signed media URL is
        # normally valid long enough to serve several adjacent playback chunks.
        self._source_cache: dict[str, tuple[float, ResolvedSource]] = {}
        self._source_cache_ttl_seconds = 600

    def _resolve_source(self, url: str, *, refresh: bool = False) -> ResolvedSource:
        now = time.monotonic()
        cached = self._source_cache.get(url)
        if not refresh and cached and now - cached[0] < self._source_cache_ttl_seconds:
            return cached[1]
        source = resolve_source(url, self.settings)
        self._source_cache[url] = (now, source)
        return source

    def _extract_job_audio(self, job: dict, audio_path: Path) -> float:
        source_url = job["source_url"]
        source = self._resolve_source(source_url)
        try:
            return extract_audio(
                source,
                audio_path,
                job["clip_start_seconds"],
                job["clip_duration_seconds"],
                self.settings.decode_timeout_seconds,
            )
        except PipelineError:
            # Signed media URLs can expire before the TTL. Refresh once instead
            # of failing a live playback chunk immediately.
            self._source_cache.pop(source_url, None)
            source = self._resolve_source(source_url, refresh=True)
            return extract_audio(
                source,
                audio_path,
                job["clip_start_seconds"],
                job["clip_duration_seconds"],
                self.settings.decode_timeout_seconds,
            )

    def _request(self, method: str, path: str, **kwargs):
        response = self.session.request(method, f"{self.server_url}{path}", timeout=900, **kwargs)
        response.raise_for_status()
        return response

    def run_once(self) -> bool:
        job = self._request("POST", "/v1/ingest/jobs/claim").json()
        if job is None:
            return False
        job_id = job["id"]
        audio_path = self.work_dir / f"{job_id}.wav"
        try:
            duration = self._extract_job_audio(job, audio_path)
            with audio_path.open("rb") as audio_file:
                self._request(
                    "PUT",
                    f"/v1/ingest/jobs/{job_id}/audio",
                    params={"duration_seconds": duration},
                    data=audio_file,
                    headers={"content-type": "audio/wav"},
                )
        except Exception as exc:
            try:
                self._request(
                    "POST",
                    f"/v1/ingest/jobs/{job_id}/fail",
                    json={"error": str(exc)},
                )
            except Exception as report_error:  # noqa: BLE001
                print(f"Could not report ingest failure: {report_error}")
            raise
        finally:
            audio_path.unlink(missing_ok=True)
        return True

    def run_forever(self, poll_seconds: float = 2) -> None:
        print(f"LGN ingest worker connected to {self.server_url}")
        while True:
            try:
                if not self.run_once():
                    time.sleep(poll_seconds)
            except KeyboardInterrupt:
                print("Ingest worker stopped")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"Ingest worker error: {exc}")
                time.sleep(5)
