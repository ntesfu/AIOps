from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.download_industreal import Archive, safe_extract, validate_archive


class IndustRealDownloadTest(unittest.TestCase):
    def test_maintenance_html_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.zip"
            path.write_text("<!DOCTYPE html><h1>Maintenance</h1>", encoding="utf-8")
            archive = Archive("train", path.name, "unused", 10_000, "unused")
            with self.assertRaisesRegex(RuntimeError, "maintenance"):
                validate_archive(path, archive, verify_md5=False)

    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "unsafe.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escape.txt", "bad")
            with self.assertRaisesRegex(RuntimeError, "unsafe archive member"):
                safe_extract(path, root / "output")


if __name__ == "__main__":
    unittest.main()
