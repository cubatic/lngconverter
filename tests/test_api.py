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


def test_browser_capture_upload(tmp_path):
    client = TestClient(create_app(Settings(artifact_dir=tmp_path)))
    home = client.get("/")
    assert home.status_code == 200
    assert "Hindi dubbing capture test" in home.text

    created = client.post("/v1/capture-sessions")
    assert created.status_code == 201
    session_id = created.json()["session_id"]

    uploaded = client.post(
        f"/v1/capture-sessions/{session_id}/chunks/0",
        content=b"webm-test-data",
        headers={"content-type": "audio/webm"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["bytes"] == 14
    assert (tmp_path / "captures" / session_id / "000000.webm").read_bytes() == b"webm-test-data"
