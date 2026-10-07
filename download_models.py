"""Download the default Irodori inference files"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODEL = "Aratako/Irodori-TTS-600M-v3-VoiceDesign"
DEFAULT_CODEC = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_TOKENIZER = "llm-jp/llm-jp-3-150m"
DEFAULT_REVISIONS = {
    DEFAULT_MODEL: "e863a3a93e652e09afeff3e84823a206a0a60314",
    DEFAULT_CODEC: "47376ee24834d7a05a48ebabfe3cde29b3c5e214",
    DEFAULT_TOKENIZER: "b112feef602fff752e4dac4c30af6a2c2fa41c7a",
}
MODEL_FILES = ["model.safetensors", "README.md", "EMOJI_ANNOTATIONS.md"]
CODEC_FILES = ["weights.pth", "README.md"]
TOKENIZER_FILES = [
    "config.json", "generation_config.json", "special_tokens_map.json",
    "tokenizer*", "vocab*", "merges.txt", "added_tokens.json", "*.model", "chat_template*",
]


def main(argv=None) -> int:
    root = Path(os.environ.get("IRODORI_PROJECT_DIR", Path(__file__).resolve().parent)).expanduser().resolve()
    default_cache = Path(os.environ.get("IRODORI_MODEL_DIR", root / "models"))
    parser = argparse.ArgumentParser(
        description="Download Irodori model, codec and tokenizer files without loading the inference runtime.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=default_cache, help="Model cache root; contains the hub directory")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model repository")
    parser.add_argument("--codec", default=DEFAULT_CODEC, help="Hugging Face codec repository")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="Hugging Face tokenizer repository")
    for component in ("model", "codec", "tokenizer"):
        parser.add_argument(f"--{component}-revision", help="Revision override; the default repository uses a pinned version, custom repositories use main")
    args = parser.parse_args(argv)

    cache = args.model_dir.expanduser().resolve()
    hub = cache / "hub"
    # Configure the cache before importing huggingface_hub, which reads these at import time.
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub)
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    from huggingface_hub import snapshot_download

    downloads = [
        ("model", args.model, args.model_revision, MODEL_FILES, "model.safetensors"),
        ("codec", args.codec, args.codec_revision, CODEC_FILES, "weights.pth"),
        ("tokenizer", args.tokenizer, args.tokenizer_revision, TOKENIZER_FILES, None),
    ]
    print(f"Model cache: {cache}", flush=True)
    for label, repo, override, files, required in downloads:
        revision = override or DEFAULT_REVISIONS.get(repo, "main")
        print(f"Downloading {label}: {repo} @ {revision}", flush=True)
        snapshot = Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=files, cache_dir=hub,
                                          local_files_only=False))
        if required and not (snapshot / required).is_file():
            raise FileNotFoundError(f"Repository {repo} does not contain the required file: {required}")
        if revision != "main":
            # The offline WebUI resolves repository IDs through this local cache ref.
            ref = snapshot.parent.parent / "refs" / "main"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(snapshot.name, encoding="utf-8")
        print(f"Ready: {snapshot}", flush=True)
    print("Model, codec and tokenizer downloads finished.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
