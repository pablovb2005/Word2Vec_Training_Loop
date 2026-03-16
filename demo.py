"""Small, reproducible demo for skip-gram with negative sampling."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from word2vec.demo import run_demo


def main() -> None:
    """Run a minimal training loop and print similarity queries."""
    corpus_path = Path("data") / "tiny_corpus.txt"
    artifact_path = Path("artifacts") / "models" / "custom_tiny_corpus.npz"
    epoch_losses, neighbors = run_demo(corpus_path, save_artifact_path=artifact_path)

    print("Epoch losses:")
    for epoch, loss in enumerate(epoch_losses, start=1):
        print(f"  epoch {epoch:02d}: {loss:.4f}")

    print("\nNearest neighbors:")
    for query, results in neighbors:
        formatted = ", ".join([f"{token} ({score:.3f})" for token, score in results])
        print(f"  {query}: {formatted}")

    print(f"\nSaved model artifact: {artifact_path}")


if __name__ == "__main__":
    main()
