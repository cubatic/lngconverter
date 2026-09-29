from fastapi.testclient import TestClient

from lgnconverter.api import create_app
from lgnconverter.config import Settings


def test_health():
    client = TestClient(create_app(Settings()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_requires_authorization_confirmation():
    client = TestClient(create_app(Settings()))
    response = client.post(
        "/v1/jobs",
        json={"source_url": "https://youtu.be/example", "authorization_confirmed": False},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Source authorization must be confirmed"


def test_youtube_is_disabled_by_default():
    client = TestClient(create_app(Settings(enable_youtube_source=False)))
    response = client.post(
        "/v1/jobs",
        json={"source_url": "https://youtu.be/example", "authorization_confirmed": True},
    )
    assert response.status_code == 422
    assert "disabled" in response.json()["detail"]


def test_browser_player_page(tmp_path):
    client = TestClient(create_app(Settings(artifact_dir=tmp_path)))
    home = client.get("/")
    assert home.status_code == 200
    assert "Hindi dubbing player" in home.text


def test_api_key_protects_job_endpoints(tmp_path):
    client = TestClient(create_app(Settings(artifact_dir=tmp_path, api_key="secret-key")))
    response = client.get("/v1/jobs/missing")
    assert response.status_code == 401
    response = client.get("/v1/jobs/missing", headers={"x-lgn-api-key": "secret-key"})
    assert response.status_code == 404


def test_remote_worker_claims_and_completes_job(tmp_path):
    client = TestClient(
        create_app(
            Settings(
                artifact_dir=tmp_path,
                api_key="client-key",
                worker_api_key="worker-key",
                enable_youtube_source=True,
                execution_mode="remote",
            )
        )
    )
    created = client.post(
        "/v1/jobs",
        headers={"x-lgn-api-key": "client-key"},
        json={
            "source_url": "https://youtu.be/authorized",
            "authorization_confirmed": True,
            "source_language": "zh",
        },
    )
    assert created.status_code == 202
    job_id = created.json()["id"]

    denied = client.post("/v1/worker/jobs/claim")
    assert denied.status_code == 401
    claimed = client.post(
        "/v1/worker/jobs/claim", headers={"x-lgn-worker-key": "worker-key"}
    )
    assert claimed.status_code == 200
    assert claimed.json()["id"] == job_id
    assert claimed.json()["status"] == "processing"

    completed = client.post(
        f"/v1/worker/jobs/{job_id}/complete",
        headers={"x-lgn-worker-key": "worker-key"},
        json={
            "detected_language": "zh",
            "duration_seconds": 2,
            "transcript": [{"start": 0, "end": 2, "text": "你好"}],
            "hindi_dialogue": [
                {
                    "start": 0,
                    "end": 2,
                    "source_text": "你好",
                    "hindi_text": "नमस्ते",
                }
            ],
        },
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"
    assert completed.json()["hindi_dialogue"][0]["hindi_text"] == "नमस्ते"
