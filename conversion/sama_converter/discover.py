#!/usr/bin/env python3
"""
Crawl a dataset's source directory and list the items to convert.
Items whose events.h5 already exists under the output root are skipped → free resumability.

  mode=frames : directories that directly contain .jpg files (one clip each)
  mode=video  : video files
  mode=images : image files
"""

from pathlib import Path

from .convert_clip import output_path


VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _ls(d: Path) -> list:
    """iterdir() with error swallowing — reliable on Lustre where rglob is not."""
    try:
        return list(d.iterdir())
    except (PermissionError, OSError):
        return []


def _find_frame_dirs(root: Path, depth: int, out: list) -> None:
    """Collect directories that directly contain .jpg files, up to `depth` levels."""
    entries = _ls(root)
    if any(e.is_file() and e.suffix.lower() == ".jpg" for e in entries):
        out.append(root)
        return
    if depth > 0:
        for e in sorted(entries):
            if e.is_dir():
                _find_frame_dirs(e, depth - 1, out)


def _find_files(root: Path, exts: set, depth: int, out: list) -> None:
    for e in sorted(_ls(root)):
        if e.is_file() and e.suffix.lower() in exts:
            out.append(e)
        elif e.is_dir() and depth > 0:
            _find_files(e, exts, depth - 1, out)


def discover_items(cfg: dict, source_root, output_root) -> list:
    """All items under source_root for cfg['mode'] whose output does not exist yet."""
    source_root = Path(source_root)
    mode        = cfg["mode"]

    raw_items = []
    if mode == "video":
        _find_files(source_root, VIDEO_EXTS, depth=3, out=raw_items)
    elif mode == "frames":
        # depth=3 covers: source_root/split/JPEGImages/clip_id  (MeViS-style)
        # and:            source_root/split/clip_id              (LV-VIS-style)
        _find_frame_dirs(source_root, depth=3, out=raw_items)
        raw_items.sort()
    elif mode == "images":
        _find_files(source_root, IMAGE_EXTS, depth=3, out=raw_items)

    pending = [src for src in raw_items if not output_path(src, cfg, output_root).exists()]
    print(f"[{cfg['dataset']}] {len(pending)} pending  |  {len(raw_items) - len(pending)} already done")
    return pending
