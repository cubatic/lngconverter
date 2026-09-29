# LGN Converter

Authorized URL-to-Hindi dubbing prototype. The current vertical slice accepts a
media URL, consumes its stream with FFmpeg without saving the original video,
and generates a timestamped faster-whisper transcript.

## Current status

- FastAPI job API
- explicit source-authorization confirmation
- direct HTTP(S) and optional YouTube source adapters
- no full source-video download; FFmpeg decodes the remote stream to mono WAV
- configurable three-minute prototype cap
- faster-whisper transcription with VAD and timestamps

Translation, stem separation, Hindi TTS, mixing, and export are the next pipeline
stages. This first slice verifies ingestion and transcription before loading the
larger models.

## Run locally or in Colab

Python 3.11 is recommended. FFmpeg must be installed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[inference,dev]'
cp .env.example .env
uvicorn lgnconverter.api:app --host 0.0.0.0 --port 8000
```

Only after verifying the required content rights and platform authorization,
set this in `.env`:

```dotenv
LGN_ENABLE_YOUTUBE_SOURCE=true
```

Create a job:

```bash
curl -X POST http://127.0.0.1:8000/v1/jobs \
  -H 'content-type: application/json' \
  -d '{
    "source_url": "https://www.youtube.com/watch?v=AUTHORIZED_VIDEO_ID",
    "authorization_confirmed": true,
    "source_language": "zh",
    "target_language": "hi"
  }'
```

Poll the returned job ID:

```bash
curl http://127.0.0.1:8000/v1/jobs/JOB_ID
```

The in-memory job store is intentionally temporary for the prototype. A durable
queue and database will replace it before multi-user deployment.

## Colab installation cell

```python
!apt-get update -qq && apt-get install -y -qq ffmpeg
!git clone YOUR_REPOSITORY_URL /content/Lgnconverter
%cd /content/Lgnconverter
!pip install -e '.[inference]'
```

Do not put account passwords, cookies, or tokens in the repository or notebook.
