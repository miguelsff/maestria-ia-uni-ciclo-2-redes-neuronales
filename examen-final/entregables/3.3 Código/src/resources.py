"""Resolve portable resources for the notebooks."""
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
import zipfile

FINAL_SHA256 = "c62eb62e651740587ba8b0c994778269e2d85f05ae55ee06312ffdf6800cf129"


def resolve_file(local: Path, url: str, destination: Path) -> Path:
    """Prefer a local file, otherwise download a direct or public Drive link."""
    if local.is_file():
        return local
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if urlparse(url).hostname in {"drive.google.com", "docs.google.com"}:
        import gdown
        if not gdown.download(url=url, output=str(destination), fuzzy=True, quiet=False):
            raise RuntimeError("No se pudo descargar el recurso de Drive")
    else:
        urlretrieve(url, destination)
    return destination


def resolve_weights(local: Path, url: str, destination: Path) -> Path:
    """Resolve and verify the published final checkpoint."""
    path = resolve_file(local, url, destination)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != FINAL_SHA256:
        raise ValueError("Los pesos no coinciden con la versión final publicada")
    return path


def resolve_dataset(local: Path, url: str, destination: Path) -> Path:
    """Prefer a local dataset; otherwise extract its public ZIP."""
    if local.is_dir():
        return local
    existing = [p for p in destination.rglob("Dataset PeruFoodNet") if p.is_dir()] if destination.exists() else []
    if len(existing) == 1:
        return existing[0]
    archive = resolve_file(destination / "perufoodnet.zip", url, destination / "perufoodnet.zip")
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("El archivo contiene una ruta fuera de la carpeta de destino")
        bundle.extractall(destination)
    matches = [p for p in destination.rglob("Dataset PeruFoodNet") if p.is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError("Se esperaba una carpeta Dataset PeruFoodNet en el ZIP")
    return matches[0]
