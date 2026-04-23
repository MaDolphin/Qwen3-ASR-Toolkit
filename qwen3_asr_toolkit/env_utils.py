import os
from pathlib import Path

from dotenv import load_dotenv


_LOADED = False


def load_project_dotenv() -> None:
    global _LOADED
    if _LOADED:
        return

    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(dotenv_path=str(cwd_env), override=False)
    else:
        # Fallback: load nearest .env that python-dotenv can find.
        load_dotenv(override=False)

    _LOADED = True


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}. Please set it in project .env.")
    return value
