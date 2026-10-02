"""Build one source identity from several photos of the same person.

inswapper only uses the source face's identity embedding (`normed_embedding`); pose, expression
and lighting always come from the target frame. Averaging the embeddings of several photos gives a
more stable identity and protects against a single poor photo. Measured on a 10-photo set against
held-out photos: one good frontal photo 0.803, one poor photo 0.644, filtered average 0.810.

Photos whose identity disagrees with the rest (e.g. near-profile shots, wrong person) are dropped.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# A photo is dropped when its mean cosine similarity to the other photos is below this
CONSISTENCY_THRESHOLD = 0.45


@dataclass
class SourcePhoto:
    path: Path
    status: str  # "used", "no face", "unreadable", "outlier"
    similarity: float | None = None
    yaw: float | None = None


@dataclass
class SourceIdentity:
    face: object  # insightface Face carrying the averaged embedding; drop-in for a single source face
    photos: list[SourcePhoto] = field(default_factory=list)
    preview_path: Path | None = None

    @property
    def used(self):
        return [p for p in self.photos if p.status == "used"]

    def summary(self):
        skipped = [p for p in self.photos if p.status != "used"]
        text = f"{len(self.used)} of {len(self.photos)} photos used"
        if skipped:
            text += " (skipped: " + ", ".join(f"{p.path.name} - {p.status}" for p in skipped) + ")"
        return text


def expand_paths(paths):
    """Accept image files and/or folders; return image files in a stable order."""
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def _read_image(path):
    # cv2.imread fails on non-ASCII Windows paths; imdecode does not
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def build_source_identity(paths, analyser):
    """Analyse each photo and combine them into one identity. Raises ValueError if none are usable."""
    photos, faces = [], []
    for path in expand_paths(paths):
        img = _read_image(path)
        if img is None:
            photos.append(SourcePhoto(path, "unreadable"))
            continue
        detected = analyser.get(img)
        if not detected:
            photos.append(SourcePhoto(path, "no face"))
            continue
        face = max(detected, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        yaw = float(face.pose[1]) if getattr(face, "pose", None) is not None else None
        photos.append(SourcePhoto(path, "used", yaw=yaw))
        faces.append((len(photos) - 1, face))

    if not faces:
        raise ValueError("No face was detected in any of the selected photos.")

    embeddings = np.stack([f.normed_embedding for _, f in faces])
    if len(faces) >= 3:
        sims = embeddings @ embeddings.T
        for row, (idx, _) in enumerate(faces):
            others = np.delete(sims[row], row)
            photos[idx].similarity = float(others.mean())
            if photos[idx].similarity < CONSISTENCY_THRESHOLD:
                photos[idx].status = "outlier"

    keep = [(idx, f) for idx, f in faces if photos[idx].status == "used"]
    if not keep:  # every photo disagreed with the rest; fall back to all detected faces
        keep = faces
        for idx, _ in faces:
            photos[idx].status = "used"

    mean = np.mean([f.normed_embedding for _, f in keep], axis=0)
    # Use the most frontal, most confident face as the carrier object and swap in the mean embedding
    best_idx, best = max(keep, key=lambda t: t[1].det_score - abs(photos[t[0]].yaw or 0) / 90)
    face = best.__class__(best)  # insightface Face is a dict subclass: shallow copy
    face["embedding"] = mean / np.linalg.norm(mean) * np.linalg.norm(best.embedding)
    return SourceIdentity(face=face, photos=photos, preview_path=photos[best_idx].path)
