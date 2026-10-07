from __future__ import annotations

import csv
import io
import json
import re
import shutil
import threading
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey, SamplingRequest, clear_cached_runtime, get_cached_runtime,
)
from irodori_tts.speaker_inversion import load_speaker_inversion_payload

DEFAULT_MODEL = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
DEFAULT_CODEC = "Aratako/Semantic-DACVAE-Japanese-32dim"
HEADERS = ["Name", "Text", "Style", "Seed"]
DISPLAY_HEADERS = ["名称", "台词", "演技", "种子"]
CLEAR_CAPTION = "[none]"


@dataclass(frozen=True)
class Settings:
    checkpoint: str = DEFAULT_MODEL
    codec: str = DEFAULT_CODEC
    device: str = "cuda"
    model_precision: str = "bf16"
    codec_precision: str = "bf16"
    source: str = "Speaker file"
    embedding: str = ""
    reference: str = ""
    lora: str = ""
    caption: str = ""
    steps: int = 40
    duration_scale: float = 1.0
    speaker_cfg: float = 5.0
    caption_cfg: float = 3.0
    text_cfg: float = 3.0
    seed: str = ""
    trim_tail: bool = False


@dataclass(frozen=True)
class Job:
    title: str
    text: str
    caption: str
    seed: int | None


def clean_path(value: str) -> str:
    return value.strip().strip('"')


def parse_seed(value: str, label: str = "Seed") -> int | None:
    value = value.strip()
    if not value:
        return None
    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if not 0 <= seed < 2**63:
        raise ValueError(f"{label} must be between 0 and {2**63 - 1}.")
    return seed


def normalize_rows(rows: list[list]) -> list[list[str]]:
    normalized = []
    for index, row in enumerate(rows, 1):
        if len(row) != len(HEADERS):
            raise ValueError(f"Row {index} must have {len(HEADERS)} columns.")
        cells = ["" if cell is None else str(cell).strip() for cell in row]
        if not any(cells):
            continue
        if not cells[1]:
            raise ValueError(f"Row {index}: Text is empty.")
        parse_seed(cells[3], f"Row {index}: Seed")
        normalized.append(cells)
    return normalized


def make_jobs(rows: list[list], settings: Settings) -> list[Job]:
    rows = normalize_rows(rows)
    if not rows:
        raise ValueError("Add text to the queue first.")
    inherited = settings.caption.strip()
    default_seed = parse_seed(settings.seed)
    jobs = []
    for title, text, caption, seed in rows:
        if caption:
            inherited = "" if caption == CLEAR_CAPTION else caption
        jobs.append(Job(title, text, inherited, parse_seed(seed) if seed else default_seed))
    return jobs


def split_text(text: str, mode: str) -> list[list[str]]:
    parts = text.splitlines() if mode == "Line" else re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [["", part.strip(), "", ""] for part in parts if part.strip()]


def import_rows(path: str, mode: str) -> list[list[str]]:
    file = Path(path)
    content = file.read_text(encoding="utf-8-sig")
    if file.suffix.lower() == ".txt":
        return split_text(content, mode)
    if file.suffix.lower() == ".json":
        payload = json.loads(content)
        rows = payload["rows"]
    elif file.suffix.lower() == ".csv":
        parsed = list(csv.reader(io.StringIO(content)))
        if not parsed or parsed[0] not in (HEADERS, DISPLAY_HEADERS):
            raise ValueError("CSV headers must be: Name,Text,Style,Seed")
        rows = parsed[1:]
    else:
        raise ValueError("Supported queue formats: TXT, CSV, JSON.")
    return normalize_rows(rows)


def export_rows(rows: list[list], directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"queue_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.json"
    path.write_text(json.dumps({"version": 1, "headers": HEADERS, "rows": normalize_rows(rows)},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def voice_choices(directory: Path) -> list[tuple[str, str]]:
    return [(path.name.removesuffix(".speaker.safetensors"), str(path))
            for path in sorted(directory.glob("*.speaker.safetensors"))]


def import_voice(source: str, directory: Path, name: str | None = None) -> str:
    src = Path(source)
    load_speaker_inversion_payload(src)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (name or src.name)
    if target.exists():
        # Don't replace existing voice
        stem = target.name.removesuffix(".speaker.safetensors")
        target = directory / f"{stem}_{uuid.uuid4().hex[:6]}.speaker.safetensors"
    shutil.copyfile(src, target)
    return str(target)


def resolve_checkpoint(value: str) -> str:
    value = clean_path(value)
    if not value:
        raise ValueError("Select a model checkpoint.")
    if value.endswith(".speaker.safetensors"):
        raise ValueError("A speaker embedding is not a model checkpoint. Select it under Speaker file.")
    if Path(value).suffix.lower() in {".safetensors", ".pt"}:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"Model checkpoint not found: {path}")
        return str(path.absolute())
    return hf_hub_download(repo_id=value, filename="model.safetensors", local_files_only=True)


def validate_voice(settings: Settings) -> None:
    if settings.source == "Speaker file":
        if not settings.embedding:
            raise ValueError("Select or import a speaker file.")
        load_speaker_inversion_payload(settings.embedding)
    elif settings.source == "Reference audio":
        if not settings.reference or not Path(settings.reference).is_file():
            raise ValueError("Upload reference audio.")
    elif settings.source != "Voice design":
        raise ValueError("Unknown voice source.")
    if settings.lora and not Path(settings.lora).is_dir():
        raise ValueError("LoRA must point to an adapter directory.")


def build_request(job: Job, settings: Settings) -> SamplingRequest:
    return SamplingRequest(
        text=job.text, caption=job.caption or None,
        ref_embed=settings.embedding if settings.source == "Speaker file" else None,
        ref_wav=settings.reference if settings.source == "Reference audio" else None,
        no_ref=settings.source == "Voice design",
        lora_adapter=settings.lora or None,
        num_steps=settings.steps, duration_scale=settings.duration_scale,
        cfg_scale_speaker=settings.speaker_cfg, cfg_scale_caption=settings.caption_cfg,
        cfg_scale_text=settings.text_cfg, cfg_min_t=0.5,
        seed=job.seed, trim_tail=settings.trim_tail,
    )


def filename(index: int, title: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip().rstrip(" .")[:70]
    return f"{index:04d}_{stem or 'audio'}.wav"


def save_audio(path: Path, audio, sample_rate: int) -> float:
    data = audio.detach().to(device="cpu").float()
    array = data.squeeze(0).numpy() if data.shape[0] == 1 else data.T.numpy()
    sf.write(str(path), array, sample_rate, subtype="PCM_16")
    return len(array) / sample_rate


class Workbench:
    def __init__(self, directory: Path):
        self.directory = directory
        self.runtime_key: RuntimeKey | None = None

    def release(self) -> None:
        clear_cached_runtime()
        self.runtime_key = None

    def get_runtime(self, settings: Settings):
        key = RuntimeKey(checkpoint=resolve_checkpoint(settings.checkpoint),
                         model_device=settings.device, codec_device=settings.device,
                         model_precision=settings.model_precision, codec_precision=settings.codec_precision,
                         codec_repo=clean_path(settings.codec))
        if self.runtime_key is not None and self.runtime_key != key:
            self.release()
        runtime, _ = get_cached_runtime(key)
        self.runtime_key = key
        return runtime

    def run(self, rows: list[list], settings: Settings, stop: threading.Event,
            continue_on_error: bool = True):
        jobs = make_jobs(rows, settings)
        validate_voice(settings)
        run_dir = self.directory / f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True)
        records = [{"index": i, **asdict(job), "status": "Pending", "file": "", "seconds": None,
                    "used_seed": None, "error": ""} for i, job in enumerate(jobs, 1)]
        messages: list[str] = []
        manifest = run_dir / "batch.json"
        archive = None

        def snapshot(status: str, phase: str):
            payload = {"settings": asdict(settings), "status": status, "jobs": records}
            temporary = manifest.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(manifest)
            return {"status": status, "phase": phase, "records": [dict(row) for row in records],
                    "files": [row["file"] for row in records if row["file"]],
                    "archive": archive, "log": "\n".join(messages[-160:]), "directory": str(run_dir)}

        yield snapshot("Loading model...", "loading")
        try:
            runtime = self.get_runtime(settings)
            if any(job.caption for job in jobs) and not runtime.model_cfg.use_caption_condition:
                raise ValueError("This model does not support styles. Clear Style or select a VoiceDesign model.")
            if settings.source != "Voice design" and not runtime.model_cfg.use_speaker_condition_resolved:
                raise ValueError("This model does not support speaker or reference conditioning.")
        except Exception as exc:
            for record in records:
                record["status"] = "Skipped"
            messages.append(f"Model error: {exc}")
            yield snapshot(f"Model error: {exc}", "error")
            return

        for index, job in enumerate(jobs):
            if stop.is_set():
                break
            record = records[index]
            record["status"] = "Generating"
            messages.append(f"[{index + 1}/{len(jobs)}] {job.title or job.text[:40]}")
            yield snapshot(f"Generating {index + 1} / {len(jobs)}", "running")
            try:
                result = runtime.synthesize(build_request(job, settings), log_fn=messages.append)
                path = run_dir / filename(index + 1, job.title)
                seconds = save_audio(path, result.audios[0], result.sample_rate)
                record.update(status="Done", file=str(path), seconds=round(seconds, 3),
                              used_seed=result.used_seed)
            except Exception as exc:
                record.update(status="Failed", error=str(exc))
                messages.append(f"Item {index + 1} failed: {exc}")
                if not continue_on_error:
                    break
            yield snapshot(f"Processed {index + 1} / {len(jobs)}", "running")

        for record in records:
            if record["status"] == "Pending":
                record["status"] = "Stopped" if stop.is_set() else "Skipped"
        completed = sum(record["status"] == "Done" for record in records)
        failed = sum(record["status"] == "Failed" for record in records)
        status = f"{'Stopped' if stop.is_set() else 'Finished'} · {completed} completed · {failed} failed"
        phase = "stopped" if stop.is_set() else "finished"
        snapshot(status, phase)
        archive_path = run_dir / "audio.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.write(manifest, "batch.json")
            for record in records:
                if record["file"]:
                    bundle.write(record["file"], Path(record["file"]).name)
        archive = str(archive_path)
        yield snapshot(status, phase)
