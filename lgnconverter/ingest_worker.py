import time
from pathlib import Path

import requests

from .config import Settings
from .pipeline import extract_audio
from .source import resolve_source


class IngestWorker:
    def __init__(self, server_url: str, ingest_key: str, settings: Settings, work_dir: Path):
        self.server_url = server_url.rstrip("/")
        self.settings = settings
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["x-lgn-ingest-key"] = ingest_key

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
            source = resolve_source(job["source_url"], self.settings)
            duration = extract_audio(
                source,
                audio_path,
                job["clip_start_seconds"],
                job["clip_duration_seconds"],
                self.settings.decode_timeout_seconds,
            )
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
