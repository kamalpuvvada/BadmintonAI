"""Fetch the pinned author-linked MIT TrackNetV3 tracking checkpoint once."""
from pathlib import Path
import argparse
import shutil
from tempfile import TemporaryDirectory
import zipfile

from badminton_proto import sha256_file


DRIVE_ID = "1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA"
ARCHIVE_SHA256 = "8de68d34ea2457e368e22947f1309143d60a6a7f9ddccd31cd526d4c4571bcc4"
CHECKPOINT_SHA256 = "df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"


def download(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "TrackNetV3_TrackNet_best.pt"
    if target.exists():
        if sha256_file(target) != CHECKPOINT_SHA256:
            raise ValueError("Existing checkpoint has a different hash; choose a fresh directory")
        return target
    with TemporaryDirectory(dir=directory) as temporary:
        archive = directory / "TrackNetV3_ckpts.zip"
        if not archive.exists():
            import gdown
            archive = Path(temporary) / "checkpoint.zip"
            gdown.download(id=DRIVE_ID, output=str(archive), quiet=False, use_cookies=False, timeout=60)
        if sha256_file(archive) != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive differs from the recorded evaluation artifact")
        staged = Path(temporary) / "tracking.pt"
        # Read just this member; never extract archive-supplied paths or InpaintNet.
        with zipfile.ZipFile(archive) as bundle, bundle.open("ckpts/TrackNet_best.pt") as source, staged.open("wb") as output:
            shutil.copyfileobj(source, output)
        if sha256_file(staged) != CHECKPOINT_SHA256:
            raise ValueError("Tracking checkpoint hash does not match")
        staged.replace(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out/models"))
    args = parser.parse_args()
    try:
        print(download(args.out))
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile, ImportError) as exc:
        parser.exit(1, f"Checkpoint download error: {exc}\n")
