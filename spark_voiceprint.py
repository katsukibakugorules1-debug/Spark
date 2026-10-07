"""Create and check SPARK's local ECAPA-TDNN speaker voiceprint.

Install dependencies with:
    python -m pip install -r requirements.txt

The pretrained model is downloaded once, then inference and comparison run
locally. Voice matching is experimental and is not secure authentication.
"""

import argparse
from pathlib import Path

import torch

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    # Compatibility with older SpeechBrain releases.
    from speechbrain.pretrained import SpeakerRecognition


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MASTER_SAMPLE = BASE_DIR / "master_voice.wav"
DEFAULT_VOICEPRINT_PATH = BASE_DIR / "master_voice.pt"
MODEL_DIR = BASE_DIR / ".cache" / "spkrec-ecapa-voxceleb"
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_THRESHOLD = 0.5

_recognizer = None


def _get_recognizer() -> SpeakerRecognition:
    """Load the pretrained speaker model once per process."""
    global _recognizer
    if _recognizer is None:
        _recognizer = SpeakerRecognition.from_hparams(
            source=MODEL_SOURCE,
            savedir=str(MODEL_DIR),
            run_opts={"device": "cpu"},
        )
    return _recognizer


def _resolve_audio_path(path: str | Path) -> Path:
    audio_path = Path(path).expanduser()
    if not audio_path.is_absolute() and not audio_path.is_file():
        audio_path = BASE_DIR / audio_path
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    return audio_path


def get_embedding(path: str | Path) -> torch.Tensor:
    """Encode an audio file as a speaker embedding."""
    audio_path = _resolve_audio_path(path)
    with torch.inference_mode():
        embedding = _get_recognizer().encode_file(str(audio_path))
    return embedding.detach().cpu()


def save_master_voiceprint(
    sample_path: str | Path = DEFAULT_MASTER_SAMPLE,
    output_path: str | Path = DEFAULT_VOICEPRINT_PATH,
) -> Path:
    """Save the enrollment recording's embedding for later comparisons."""
    sample = _resolve_audio_path(sample_path)
    destination = Path(output_path).expanduser()
    if not destination.is_absolute():
        destination = BASE_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    embedding = get_embedding(sample)
    torch.save(embedding, destination)
    print(f"Master voiceprint saved to {destination}")
    return destination


def _load_master_embedding(
    voiceprint_path: str | Path = DEFAULT_VOICEPRINT_PATH,
) -> torch.Tensor:
    path = Path(voiceprint_path).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Voiceprint not found at {path}. "
            "Run save_master_voiceprint('master_voice.wav') first."
        )
    try:
        embedding = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch versions do not expose weights_only.
        embedding = torch.load(path, map_location="cpu")
    if not isinstance(embedding, torch.Tensor):
        raise ValueError("Saved voiceprint is not a PyTorch tensor.")
    return embedding


def similarity_score(
    new_audio_path: str | Path,
    voiceprint_path: str | Path = DEFAULT_VOICEPRINT_PATH,
) -> float:
    """Return cosine similarity between a sample and the saved voiceprint."""
    saved_embedding = _load_master_embedding(voiceprint_path)
    new_embedding = get_embedding(new_audio_path)
    with torch.inference_mode():
        score = _get_recognizer().similarity(saved_embedding, new_embedding)
    return float(score.reshape(-1)[0].item())


def is_authorized_speaker(
    new_audio_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    voiceprint_path: str | Path = DEFAULT_VOICEPRINT_PATH,
) -> bool:
    """Return whether the sample exceeds the configured similarity threshold."""
    return similarity_score(new_audio_path, voiceprint_path) > threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enroll",
        type=Path,
        default=DEFAULT_MASTER_SAMPLE,
        help="Audio clip to save as the master voiceprint.",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="Optional audio clip to compare to the saved voiceprint.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    if args.verify:
        score = similarity_score(args.verify)
        print(f"Cosine similarity: {score:.3f}")
        print("MATCH" if score > args.threshold else "NO MATCH")
    else:
        save_master_voiceprint(args.enroll)


if __name__ == "__main__":
    main()
