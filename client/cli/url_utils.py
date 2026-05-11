from urllib.parse import urlsplit, urlunsplit

DEFAULT_SERVER_URL = "http://127.0.0.1:10012"
OFFLINE_TRANSCRIBE_PATH = "/api/v1/offline/transcribe"
STREAM_WS_PATH = "/ws/stream"


def normalize_server_url(server: str | None, default: str = DEFAULT_SERVER_URL) -> str:
    value = (server or "").strip() or default
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported server URL scheme: {parsed.scheme}")
    if not parsed.netloc:
        raise ValueError(f"Invalid server URL: {server}")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def build_offline_api_url(server: str | None) -> str:
    base = normalize_server_url(server)
    return f"{base}{OFFLINE_TRANSCRIBE_PATH}"


def build_stream_ws_url(server: str | None) -> str:
    base = normalize_server_url(server)
    parsed = urlsplit(base)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunsplit((scheme, parsed.netloc, f"{parsed.path.rstrip('/')}{STREAM_WS_PATH}", "", ""))
