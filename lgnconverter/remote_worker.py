import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .dubbing import create_dubbed_audio
from .pipeline import transcribe
from .translation import translate_to_hindi


class RemoteWorker:
    def __init__(self, server_url: str, worker_key: str, settings: Settings, work_dir: Path):
        self.server_url = server_url.rstrip("/")
        self.settings = settings
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["x-lgn-worker-key"] = worker_key
        self.session.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    connect=3,
                    read=3,
                    backoff_factor=1,
                    allowed_methods={"GET"},
                )
            ),
        )

    def _request(self, method: str, path: str, **kwargs):
        response = self.session.request(method, f"{self.server_url}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response

    def run_once(self) -> bool:
        job = self._request("POST", "/v1/worker/jobs/claim").json()
        if job is None:
            return False
        job_id = job["id"]
        audio_path = self.work_dir / f"{job_id}.wav"
        timings: dict[str, float] = {}
        try:
            stage_started = time.monotonic()
            audio = self._request("GET", f"/v1/worker/jobs/{job_id}/audio").content
            audio_path.write_bytes(audio)
            timings["download"] = round(time.monotonic() - stage_started, 3)
            stage_started = time.monotonic()
            transcript, detected_language = transcribe(
                audio_path,
                self.settings,
                job.get("source_language"),
            )
            timings["transcription"] = round(time.monotonic() - stage_started, 3)
            current = self._request("GET", f"/v1/worker/jobs/{job_id}").json()
            if current["status"] in {"paused", "cancelled"}:
                return True
            stage_started = time.monotonic()
            hindi_dialogue = translate_to_hindi(
                transcript,
                job.get("source_language") or detected_language,
                self.settings,
            )
            timings["translation"] = round(time.monotonic() - stage_started, 3)
            if self.settings.enable_tts:
                dubbed_path = self.work_dir / f"{job_id}-hindi.m4a"
                timings.update(create_dubbed_audio(
                    audio_path, hindi_dialogue, dubbed_path, self.settings
                ))
                stage_started = time.monotonic()
                with dubbed_path.open("rb") as dubbed_file:
                    self._request(
                        "PUT",
                        f"/v1/worker/jobs/{job_id}/dubbed-audio",
                        data=dubbed_file,
                        headers={"content-type": "audio/mp4"},
                    )
                timings["upload"] = round(time.monotonic() - stage_started, 3)
            payload = {
                "detected_language": detected_language,
                "duration_seconds": job.get("duration_seconds") or 0,
                "transcript": [segment.model_dump() for segment in transcript],
                "hindi_dialogue": [segment.model_dump() for segment in hindi_dialogue],
                "stage_timings": timings,
            }
            self._request("POST", f"/v1/worker/jobs/{job_id}/complete", json=payload)
            print(f"Completed {job_id}: {timings}")
        except Exception as exc:
            try:
                self._request(
                    "POST",
                    f"/v1/worker/jobs/{job_id}/fail",
                    json={"error": str(exc)},
                )
            except Exception as report_error:  # noqa: BLE001
                print(f"Could not report job failure: {report_error}")
            raise
        finally:
            audio_path.unlink(missing_ok=True)
            (self.work_dir / f"{job_id}-hindi.m4a").unlink(missing_ok=True)
        return True

    def run_forever(self, poll_seconds: float = 2) -> None:
        print(f"LGN worker connected to {self.server_url}")
        while True:
            try:
                worked = self.run_once()
                if not worked:
                    time.sleep(poll_seconds)
            except KeyboardInterrupt:
                print("Worker stopped")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"Worker error: {exc}")
                time.sleep(5)
