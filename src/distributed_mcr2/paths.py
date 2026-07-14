"""Path helpers for installed and source-tree execution."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_ROOT / "resources"


def resolve_input_path(path: str | Path) -> Path:
    """Resolve an input path from absolute path, working directory, or package resources."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    resource_candidate = RESOURCE_DIR / candidate
    if resource_candidate.exists():
        return resource_candidate
    return cwd_candidate


def resolve_output_path(path: str | Path) -> Path:
    """Resolve output and dataset paths relative to the current working directory."""
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate
