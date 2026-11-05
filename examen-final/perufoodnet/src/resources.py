"""Resolve a local dataset or download its public ZIP when explicitly needed."""
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import zipfile


def resolve_dataset(local: Path, url: str, destination: Path) -> Path:
    """Prefer a local dataset; otherwise extract the configured public ZIP."""
    if local.is_dir():
        return local
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "perufoodnet.zip"
    if urlparse(url).hostname in {"drive.google.com", "docs.google.com"}:
        import gdown
        gdown.download(url=url, output=str(archive), fuzzy=True, quiet=False)
    else:
        urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    matches = [path for path in destination.rglob("Dataset PeruFoodNet") if path.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError("Expected exactly one Dataset PeruFoodNet folder in the archive")
    return matches[0]
