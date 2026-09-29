import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

from .config import Settings

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}


class SourceError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedSource:
    media_url: str
    title: str | None = None
    duration: float | None = None
    http_headers: dict[str, str] | None = None


def _is_youtube_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return host in YOUTUBE_HOSTS or host.endswith(".youtube.com")


def _assert_public_host(host: str) -> None:
    try:
        addresses = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SourceError("Source hostname could not be resolved") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise SourceError("Private, loopback, and link-local source addresses are not allowed")


def validate_source_url(url: str, settings: Settings) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SourceError("Only public HTTP(S) media URLs are supported")

    is_youtube = _is_youtube_host(parsed.hostname)
    if is_youtube and not settings.enable_youtube_source:
        raise SourceError(
            "YouTube source support is disabled. Enable it only with the required authorization."
        )
    if not is_youtube:
        _assert_public_host(parsed.hostname)
    return is_youtube


def resolve_source(url: str, settings: Settings) -> ResolvedSource:
    is_youtube = validate_source_url(url, settings)
    if not is_youtube:
        return ResolvedSource(media_url=url)

    try:
        import yt_dlp

        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "format": "bestaudio/best",
            "noplaylist": True,
        }
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
    except Exception as exc:
        raise SourceError(f"YouTube stream resolution failed: {exc}") from exc

    duration = info.get("duration")
    if duration and duration > settings.max_source_seconds:
        raise SourceError(
            f"Source is {duration:.0f}s; prototype limit is {settings.max_source_seconds}s"
        )
    media_url = info.get("url")
    if not media_url:
        raise SourceError("No playable audio stream was found")
    return ResolvedSource(
        media_url=media_url,
        title=info.get("title"),
        duration=duration,
        http_headers=info.get("http_headers"),
    )

