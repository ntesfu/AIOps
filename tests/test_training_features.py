from pathlib import Path

from aiops.training.simple_video_classifier import extract_video_features


def test_missing_or_empty_video_returns_fixed_feature_vector(tmp_path: Path) -> None:
    feature = extract_video_features(tmp_path / "missing.mp4")

    assert feature.shape == (14,)
