from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.extract_industreal_auxiliary import extract_auxiliary


class IndustRealAuxiliaryTest(unittest.TestCase):
    def test_selective_extraction_preserves_recording_and_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = root / "archives"
            archives.mkdir()
            with zipfile.ZipFile(archives / "train_p1.zip", "w") as archive:
                archive.writestr("01_assy_0_1/hands.csv", "000000.jpg,1,2\n")
                archive.writestr("01_assy_0_1/OD_labels.json", json.dumps({"images": []}))
                archive.writestr("01_assy_0_1/rgb/000000.jpg", b"large image")
            output = root / "aux"
            report = extract_auxiliary(archives, output)
            self.assertEqual(report["extracted_files"], 2)
            self.assertTrue(
                (output / "recordings/train/01_assy_0_1/hands.csv").is_file()
            )
            self.assertFalse((output / "recordings/train/01_assy_0_1/rgb").exists())
