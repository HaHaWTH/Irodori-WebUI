from __future__ import annotations

import tts_bootstrap
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from irodori_tts.codec import DACVAECodec
from irodori_tts.inference_runtime import resolve_runtime_device, resolve_runtime_dtype
from irodori_tts.text_normalization import normalize_text


def encode_samples(samples, output: Path, codec, max_seconds: float):
    latent_dir = output / "latents"
    latent_dir.mkdir()
    partial = output / "manifest.jsonl.part"
    with partial.open("w", encoding="utf-8") as manifest:
        for index, item in enumerate(samples, 1):
            audio, sample_rate = sf.read(item["audio"], dtype="float32", always_2d=True)
            if len(audio) == 0 or len(audio) / sample_rate > max_seconds:
                raise ValueError(f"Invalid audio duration: {item['audio']}")
            waveform = torch.from_numpy(audio.T.copy())
            with torch.inference_mode():
                latent = codec.encode_waveform(waveform, sample_rate=sample_rate)[0].float().cpu()
            path = latent_dir / f"{index:08d}.pt"
            torch.save(latent, path)
            payload = {"text": normalize_text(item["text"]), "latent_path": f"latents/{path.name}",
                       "num_frames": int(latent.shape[0])}
            if item["speaker_id"]:
                payload["speaker_id"] = item["speaker_id"]
            if item["caption"]:
                payload["caption"] = item["caption"]
            manifest.write(json.dumps(payload, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"[{index}/{len(samples)}] Encoded {Path(item['audio']).name}: {latent.shape[0]} frames", flush=True)
    partial.replace(output / "manifest.jsonl")
    print(f"Prepared {len(samples)} samples: {output / 'manifest.jsonl'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare local Irodori training data.")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="fp32")
    parser.add_argument("--max-seconds", type=float, default=30)
    args = parser.parse_args()
    samples = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    if not samples:
        raise ValueError("No training samples provided.")
    device = resolve_runtime_device(args.device)
    dtype = resolve_runtime_dtype(precision=args.precision, device=device)
    codec = DACVAECodec.load(repo_id=args.codec, device=str(device), dtype=dtype)
    encode_samples(samples, Path(args.output), codec, args.max_seconds)


if __name__ == "__main__":
    main()
