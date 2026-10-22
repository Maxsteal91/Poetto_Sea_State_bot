"""Salvataggio delle clip analizzate e dei voti degli utenti, per riaddestrare il modello.

Layout su disco (dentro data/, che su Railway è un volume):
    data/feedback_pending/<analysis_id>/   clip + metadata.json in attesa di voto
    data/feedback/<label>/<analysis_id>/   clip votate, una cartella per classe
    data/feedback/labels.csv               una riga per voto
Le clip non votate vengono cancellate dopo PENDING_MAX_AGE_HOURS.
"""

import csv
import datetime
import json
import logging
import re
import shutil
import time
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

PENDING_DIR = Path("data") / "feedback_pending"
LABELLED_DIR = Path("data") / "feedback"
LABELS_CSV = LABELLED_DIR / "labels.csv"
PENDING_MAX_AGE_HOURS = 24

# Formato di new_analysis_id: valida anche gli id che arrivano dai callback, prima di usarli come percorsi
ANALYSIS_ID_PATTERN = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")
LABELS_CSV_COLUMNS = [
    "analysis_id",
    "user_label",
    "predicted_label",
    "top_class",
    "confidence",
    "wave_height",
    "created_at",
    "labelled_at",
]


def new_analysis_id(now: datetime.datetime) -> str:
    """Id univoco e ordinabile per un'analisi, es. '20261022_160512_a1b2c3'"""
    return f"{now:%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def save_pending(analysis_id: str, video_paths: list[str], metadata: dict) -> Path:
    """Copia le clip analizzate e i metadati in PENDING_DIR/<analysis_id>/ in attesa del voto"""
    if not ANALYSIS_ID_PATTERN.match(analysis_id):
        raise ValueError(f"Invalid analysis_id: {analysis_id!r}")
    pending_dir = PENDING_DIR / analysis_id
    pending_dir.mkdir(parents=True)
    for index, video_path in enumerate(video_paths):
        shutil.copy2(video_path, pending_dir / f"segment_{index}{Path(video_path).suffix}")
    metadata_path = pending_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Saved {len(video_paths)} segments for feedback in {pending_dir}")
    return pending_dir


def append_label_row(row: dict) -> None:
    """Aggiunge una riga a LABELS_CSV, scrivendo l'intestazione se il file è nuovo"""
    is_new_file = not LABELS_CSV.exists()
    with LABELS_CSV.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LABELS_CSV_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)


def record_feedback(analysis_id: str, user_label: str, valid_labels: list[str], now: datetime.datetime) -> bool:
    """Sposta le clip votate in LABELLED_DIR/<user_label>/ e registra il voto in LABELS_CSV.

    Restituisce False se le clip non sono più in attesa (già votate o scadute).
    Solleva ValueError se analysis_id o user_label non sono validi.
    """
    if not ANALYSIS_ID_PATTERN.match(analysis_id):
        raise ValueError(f"Invalid analysis_id: {analysis_id!r}")
    if user_label not in valid_labels:
        raise ValueError(f"Invalid label {user_label!r}, expected one of {valid_labels}")

    pending_dir = PENDING_DIR / analysis_id
    metadata_path = pending_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    target_dir = LABELLED_DIR / user_label / analysis_id
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(pending_dir), str(target_dir))
    except FileNotFoundError:
        # Two votes on the same result at the same time: the first one wins
        return False

    append_label_row(
        {
            "analysis_id": analysis_id,
            "user_label": user_label,
            "predicted_label": metadata.get("predicted_label"),
            "top_class": metadata.get("top_class"),
            "confidence": metadata.get("confidence"),
            "wave_height": metadata.get("wave_height"),
            "created_at": metadata.get("created_at"),
            "labelled_at": now.isoformat(timespec="seconds"),
        }
    )
    log.info(f"Feedback {analysis_id}: user says {user_label}, model said {metadata.get('predicted_label')}")
    return True


def delete_expired_pending(max_age_hours: float = PENDING_MAX_AGE_HOURS) -> int:
    """Cancella le cartelle in attesa di voto più vecchie di max_age_hours; restituisce quante"""
    if not PENDING_DIR.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    deleted = 0
    for pending_dir in PENDING_DIR.iterdir():
        if pending_dir.is_dir() and pending_dir.stat().st_mtime < cutoff:
            shutil.rmtree(pending_dir)
            deleted += 1
    return deleted
