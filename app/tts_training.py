from __future__ import annotations

import atexit
import csv
import json
import math
import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import soundfile as sf
import yaml
from safetensors import safe_open

from tts_bootstrap import DATA, RUNTIME
from tts_engine import clean_path, resolve_checkpoint
from irodori_tts.config import ModelConfig, TrainConfig

GPU_LOCK = threading.Lock()
AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3"}
INFERENCE_KEYS = {"max_text_len", "max_caption_len", "fixed_target_latent_steps"}


def read_samples(directory: str, table: str = "", speaker: str = "speaker", caption: str = "",
                 max_seconds: float = 30.0) -> list[dict]:
    root = Path(clean_path(directory)).expanduser().resolve()
    if not directory.strip() or not root.is_dir():
        raise ValueError("Select an existing audio directory.")
    if max_seconds <= 0 or not math.isfinite(max_seconds):
        raise ValueError("Maximum duration must be a positive number.")
    entries = []
    if table.strip():
        with Path(clean_path(table)).expanduser().open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"file_name", "text"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV requires file_name and text columns; caption and speaker_id are optional.")
            for row in reader:
                if not any(row.values()):
                    continue
                file_name = (row["file_name"] or "").strip()
                if not file_name:
                    raise ValueError(f"CSV line {reader.line_num}: file_name is empty.")
                audio = Path(file_name).expanduser()
                if not audio.is_absolute():
                    audio = root / audio
                entries.append({"audio": str(audio.resolve()), "text": (row["text"] or "").strip(),
                                "caption": (row.get("caption") or caption).strip(),
                                "speaker_id": (row.get("speaker_id") or speaker).strip()})
    else:
        for audio in sorted(path for path in root.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES and path.is_file()):
            transcript = audio.with_suffix(".txt")
            if not transcript.is_file():
                raise ValueError(f"Missing transcript: {transcript}")
            entries.append({"audio": str(audio.resolve()), "text": transcript.read_text(encoding="utf-8-sig").strip(),
                            "caption": caption.strip(), "speaker_id": speaker.strip()})
    if not entries:
        raise ValueError("No training samples found.")
    seen = set()
    for index, item in enumerate(entries, 1):
        path = Path(item["audio"])
        if path in seen:
            raise ValueError(f"Duplicate audio in dataset: {path}")
        seen.add(path)
        if not item["text"]:
            raise ValueError(f"Sample {index}: transcript is empty.")
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        if info.frames == 0:
            raise ValueError(f"Empty audio: {path}")
        if duration > max_seconds:
            raise ValueError(f"Audio exceeds {max_seconds:g} seconds: {path} ({duration:.2f}s). Split audio and text first.")
        item.update(seconds=round(duration, 3), sample_rate=info.samplerate)
    return entries


def read_model_config(checkpoint: str) -> tuple[str, dict]:
    path = Path(resolve_checkpoint(checkpoint))
    if path.suffix.lower() == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if "config_json" not in metadata:
            raise ValueError("The checkpoint does not contain model configuration metadata.")
        flat = json.loads(metadata["config_json"])
        model = {key: value for key, value in flat.items() if key not in INFERENCE_KEYS}
    else:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = payload.get("model_config")
        if model is None:
            raise ValueError("The checkpoint does not contain model_config.")
    ModelConfig(**model)
    return str(path.resolve()), model


def validate_manifest(value: str, max_frames: int) -> Path:
    if not value.strip():
        raise ValueError("Select a training manifest.")
    path = Path(clean_path(value)).expanduser().resolve()
    count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not str(item.get("text", "")).strip() or not item.get("latent_path"):
                raise ValueError(f"Manifest line {line_number} requires text and latent_path.")
            latent = Path(item["latent_path"])
            if not latent.is_absolute():
                latent = path.parent / latent
            if not latent.is_file():
                raise ValueError(f"Latent file not found: {latent}")
            if item.get("num_frames", 0) > max_frames:
                raise ValueError(f"Manifest line {line_number} exceeds the maximum latent frames. Increase the limit or split the sample.")
            count += 1
    if not count:
        raise ValueError("Training manifest is empty.")
    return path


@dataclass(frozen=True)
class TrainingSettings:
    checkpoint: str
    manifest: str
    mode: str = "speaker"
    device: str = "cuda"
    precision: str = "bf16"
    name: str = "voice"
    output_root: str = ""
    steps: int = 3000
    batch_size: int = 1
    accumulation: int = 4
    learning_rate: float = .01
    save_every: int = 250
    gradient_checkpointing: bool = True
    max_frames: int = 750
    tokens: int = 16
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_target: str = "diffusion_attn"
    continuation: str = ""
    seed: int = 0


def training_config(settings: TrainingSettings) -> tuple[str, Path, dict]:
    if settings.mode not in {"speaker", "lora"}:
        raise ValueError("Unknown training mode.")
    for label, value in (("Steps", settings.steps), ("Batch size", settings.batch_size),
                         ("Accumulation", settings.accumulation), ("Save interval", settings.save_every),
                         ("Maximum frames", settings.max_frames), ("Speaker tokens", settings.tokens),
                         ("LoRA rank", settings.lora_rank), ("LoRA alpha", settings.lora_alpha)):
        if value < 1:
            raise ValueError(f"{label} must be positive.")
    if not math.isfinite(settings.learning_rate) or settings.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    checkpoint, model = read_model_config(settings.checkpoint)
    if settings.mode == "speaker" and not ModelConfig(**model).use_speaker_condition_resolved:
        raise ValueError("Voice training requires a speaker-conditioned base model.")
    manifest = validate_manifest(settings.manifest, settings.max_frames)
    speaker = settings.mode == "speaker"
    train = {
        "train_mode": "rf", "precision": settings.precision, "batch_size": settings.batch_size,
        "gradient_accumulation_steps": settings.accumulation, "num_workers": 0,
        "dataloader_persistent_workers": False, "compile_model": False,
        "gradient_checkpointing": settings.gradient_checkpointing, "optimizer": "adamw",
        "learning_rate": settings.learning_rate, "weight_decay": 0.0 if speaker else .01,
        "lr_scheduler": "none", "max_steps": settings.steps, "log_every": 10,
        "save_every": settings.save_every, "checkpoint_best_n": 0, "valid_ratio": 0.0,
        "valid_every": 0, "wandb_enabled": False, "progress": True, "seed": settings.seed,
        "max_text_len": 256, "max_caption_len": 512, "max_latent_steps": settings.max_frames,
        "fixed_target_latent_steps": None, "fixed_target_full_mask": False, "rf_loss_mode": "utterance_mean",
        "text_condition_dropout": 0.0 if speaker else .1,
        "speaker_condition_dropout": 0.0 if speaker else .1,
        "caption_condition_dropout": 0.0 if speaker else .1,
        "duration_speaker_dropout": 0.0 if speaker else .1,
        "duration_caption_dropout": 0.0 if speaker else .1,
        "speaker_inversion_enabled": speaker, "speaker_inversion_tokens": settings.tokens,
        "lora_enabled": not speaker, "lora_r": settings.lora_rank, "lora_alpha": settings.lora_alpha,
        "lora_target_modules": settings.lora_target, "lora_modules_to_save": "none",
    }
    continuation = clean_path(settings.continuation)
    if continuation:
        path = Path(continuation).expanduser().resolve()
        if speaker:
            from irodori_tts.speaker_inversion import load_speaker_inversion_payload
            embedding = load_speaker_inversion_payload(path)["speaker_embedding"]
            if tuple(embedding.shape) != (settings.tokens, ModelConfig(**model).speaker_dim):
                raise ValueError("Continuation embedding does not match the selected token count and speaker dimension.")
            train["speaker_inversion_init_embedding"] = str(path)
        elif not (path.is_dir() and (path / "trainer_state.pt").is_file()):
            raise ValueError("LoRA continuation requires a checkpoint directory containing trainer_state.pt.")
    TrainConfig(**train)
    return checkpoint, manifest, {"model": model, "train": train}


@dataclass
class Job:
    id: str
    kind: str
    directory: Path
    log: Path
    command: list[str]
    mode: str = ""
    checkpoint: str = ""
    status: str = "Running"
    returncode: int | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)
    stopped: bool = False


class JobManager:
    def __init__(self, lock=GPU_LOCK):
        self.lock = lock
        self.jobs: dict[str, Job] = {}
        self._state_lock = threading.Lock()

    def start(self, kind, name, output_root, configure, release_model):
        if not self.lock.acquire(blocking=False):
            raise ValueError("The GPU is busy. Wait for the current task or stop training first.")
        try:
            release_model()
            slug = re.sub(r"[^\w-]+", "_", name.strip()).strip("_") or "voice"
            job_id = f"{slug}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
            root = Path(clean_path(output_root)).expanduser().resolve()
            directory = root / job_id
            directory.mkdir(parents=True)
            command, details = configure(directory)
            job = Job(job_id, kind, directory, directory / "run.log", command, **details)
            (directory / "job.json").write_text(json.dumps({"kind": kind, "command": command, **details},
                                                          ensure_ascii=False, indent=2), encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONUTF8"] = "1"
            with job.log.open("wb") as output:
                job.process = subprocess.Popen(command, cwd=RUNTIME, env=env, stdout=output,
                                                stderr=subprocess.STDOUT,
                                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            with self._state_lock:
                self.jobs[job_id] = job
            threading.Thread(target=self._watch, args=(job,), daemon=True).start()
            return job
        except Exception:
            self.lock.release()
            raise

    def _watch(self, job):
        try:
            code = job.process.wait()
            with self._state_lock:
                job.returncode = code
                job.status = "Stopped" if job.stopped else ("Done" if code == 0 else "Failed")
        finally:
            self.lock.release()

    def get(self, job_id):
        with self._state_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise ValueError("Job not found in this WebUI process.")
            return job

    def latest(self):
        with self._state_lock:
            return next(reversed(self.jobs.values()), None)

    def stop(self, job_id):
        job = self.get(job_id)
        with self._state_lock:
            if job.process.poll() is None:
                job.stopped = True
                job.process.terminate()

    def shutdown(self):
        for job in list(self.jobs.values()):
            if job.process.poll() is None:
                job.process.terminate()
                try:
                    job.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    job.process.kill()
                    job.process.wait()


JOBS = JobManager()
atexit.register(JOBS.shutdown)


def log_tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 24000))
        return stream.read().decode("utf-8", errors="replace")


def artifacts(job: Job) -> list[str]:
    if job.kind == "prepare":
        manifest = job.directory / "manifest.jsonl"
        return [str(manifest)] if manifest.is_file() else []
    if job.mode == "speaker":
        found = list(job.directory.glob("*.speaker.safetensors"))
    else:
        found = [path.parent for path in job.directory.glob("*/adapter_config.json")]
    return [str(path) for path in sorted(found, key=lambda p: ("final" not in p.name, p.name))]


def start_preparation(name, directory, table, speaker, caption, max_seconds,
                      codec, device, precision, output_root, release_model):
    samples = read_samples(directory, table, speaker, caption, float(max_seconds))
    def configure(destination):
        source = destination / "samples.json"
        source.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
        command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).with_name("prepare_training.py")),
                   "--samples", str(source), "--output", str(destination), "--codec", codec,
                   "--device", device, "--precision", precision, "--max-seconds", str(max_seconds)]
        return command, {}
    return JOBS.start("prepare", name, output_root or str(DATA / "training" / "datasets"), configure, release_model)


def start_training(settings, release_model):
    checkpoint, manifest, config = training_config(settings)
    def configure(destination):
        config_path = destination / "training.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-u", "-X", "utf8", str(RUNTIME / "train.py"), "--config", str(config_path),
                   "--manifest", str(manifest), "--init-checkpoint", checkpoint,
                   "--output-dir", str(destination), "--device", settings.device]
        if settings.mode == "lora" and settings.continuation.strip():
            command += ["--resume", str(Path(clean_path(settings.continuation)).expanduser().resolve())]
        return command, {"mode": settings.mode, "checkpoint": checkpoint}
    return JOBS.start("train", settings.name, settings.output_root or str(DATA / "training" / "runs"), configure, release_model)
