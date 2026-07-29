"""Central configuration for the InVision dataset pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "Images"
VIDEOS_DIR = DATA_DIR / "Videos"
DERIVED_DIR = DATA_DIR / "derived"

# Model identifiers. CLIP weights (~600MB) download from HuggingFace on first run.
CLIP_MODEL = "openai/clip-vit-base-patch32"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}

# A store needs at least this many images to contribute both gallery and query rows.
MIN_IMAGES_PER_STORE = 3

# Images held out per store for evaluation. Never encoded into the gallery.
QUERY_PER_STORE = 1

# Capture folders are named by ordinal, but the mall's "1st" is the ground
# floor, so every ordinal maps one level down. Folder ordinal -> real floor.
FOLDER_ORDINAL_TO_FLOOR = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

FLOOR_DISPLAY_NAMES = {0: "Ground", 1: "First", 2: "Second", 3: "Third", 4: "Fourth"}

# "1st Left", "2nd right", "3rd Left" -> (ordinal, side)
FOLDER_PATTERN = r"^(\d+)(?:st|nd|rd|th)\s+(left|right)$"

# Capture timestamp embedded in the filename: 20260709_161752.jpg
TIMESTAMP_PATTERN = r"(\d{8})_(\d{6})"

# The floor directory is the source of truth for navigation, so every
# captured corridor is ingested however sparse. Sparse floors simply give
# weaker recognition; they remain routable destinations.
MIN_IMAGES_PER_FLOOR_SIDE = 1

# Adjacent-frame cosine similarity at or above this keeps two consecutive
# shots in the same store group. Calibrated in tools/calibrate_threshold.py:
# 0.80 is the only value that holds both hand-verified bursts together while
# still splitting them from their neighbours. The margin is thin (the tightest
# true pair scores 0.812), so this is a fallback signal only - OCR brand
# agreement is the primary grouping evidence.
GROUPING_SIMILARITY = 0.80

# OCR tokens shorter than this, or purely numeric, are treated as promotional
# noise rather than brand names.
MIN_BRAND_TOKEN_LEN = 3
