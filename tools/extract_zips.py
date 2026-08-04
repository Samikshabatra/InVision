"""Extract the captured floor/side zips into data/Images/.

Each zip contains a single top-level folder (e.g. "1st Left") holding a flat
set of timestamped photographs, so the archives are unpacked as-is.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEST = PROJECT_ROOT / "data" / "Images"

# Strip the Google Drive export suffix: "1st Left-20260728T143813Z-1-001.zip"
DRIVE_SUFFIX = re.compile(r"-\d{8}T\d{6}Z-\d+-\d+$")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    zips = sorted(PROJECT_ROOT.glob("*.zip"))
    if not zips:
        print("no zips found")
        return

    for archive in zips:
        label = DRIVE_SUFFIX.sub("", archive.stem)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            zf.extractall(DEST)
        print(f"{label:<12} {len(members):>4} files")

    print(f"\nextracted into {DEST}")


if __name__ == "__main__":
    main()
