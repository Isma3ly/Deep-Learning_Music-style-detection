import argparse
import sys
from pathlib import Path

from .config import DEFAULT_FILLER_TRACK
from .inference import predict_genre_from_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Music genre classifier CLI")
    parser.add_argument(
        "audio_path",
        help="Path to the WAV file to classify",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing DNN3.weights.h5 and pca (default: models)",
    )
    parser.add_argument(
        "--filler-track",
        default=None,
        help="Path to the filler track used when fitting PCA (default: data/genres_original/rock/rock.00000.wav)",
    )
    args = parser.parse_args(argv)

    models_dir = Path(args.models_dir)
    weights_path = models_dir / "DNN3.weights.h5"
    pca_path = models_dir / "pca"

    if not weights_path.exists() or not pca_path.exists():
        print("Error: Model files not found. Run the training notebook (notebooks/02_train_model.ipynb) first, or set --models-dir to the folder containing DNN3.weights.h5 and pca.", file=sys.stderr)
        sys.exit(1)

    filler = args.filler_track if args.filler_track is not None else DEFAULT_FILLER_TRACK
    genre, prob, conf_text = predict_genre_from_file(
        args.audio_path,
        model_weights=str(weights_path),
        pca_path=str(pca_path),
        filler_track_path=filler,
    )
    print(f"{conf_text}{genre} (p={prob:.2f})")


if __name__ == "__main__":
    main()

