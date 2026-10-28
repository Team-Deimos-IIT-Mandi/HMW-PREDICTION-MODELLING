"""Make predictions from a CSV using the saved best model."""
import argparse
from pathlib import Path

import pandas as pd

from src.inference import predict_frame
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(f"Prediction input not found: {source}")
    frame = pd.read_csv(source)
    name, predictions = predict_frame(frame, config)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} predictions with {name} to {args.output}")


if __name__ == "__main__":
    main()
