"""
Format trajectory dataset:
    - Crop trajectories greater than a certain length;
    - Gather all chunk information (traj, annotations, character, etc...).
"""

import argparse
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from decord import VideoReader, cpu

from helper.files import load_txt
from helper.progress import PROGRESS

# ------------------------------------------------------------------------------------- #

num_frames, num_vertices, num_faces, width, height = None, None, None, None, None
MAX_NUM_FRAMES = 300

# ------------------------------------------------------------------------------------- #


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "cm_dir",
        type=Path,
        help="Path the CondensedMovies dataset directory",
    )
    parser.add_argument(
        "et_dir",
        type=Path,
        help="Path the E.T. dataset directory",
    )

    args = parser.parse_args()

    return args.__dict__


# ------------------------------------------------------------------------------------- #


def load_shot(input_dir: Path, year_name: str, clip_name: str) -> Dict[int, List[int]]:
    shot_dir = input_dir / "shots" / "raw"
    shot_path = shot_dir / year_name / clip_name / "shot_txt" / (clip_name + ".txt")

    if shot_path.exists():
        shot_txt = load_txt(shot_path)
        shot_to_bounds = {}
        for shot_index, shot in enumerate(shot_txt.split("\n")):
            shot_boundaries = [int(t) for t in shot.split(" ")[:2]]
            shot_to_bounds[shot_index] = shot_boundaries
        return shot_to_bounds

    else:
        shot_to_bounds = None


def load_frames(video_path: Path, start_frames: int, end_frames: int) -> np.ndarray:
    try:
        with open(video_path, "rb") as f:
            vr = VideoReader(f, ctx=cpu(0))
        frames = vr.get_batch(range(start_frames, end_frames + 1, 1))
        return frames.asnumpy()
    except RuntimeError:
        return np.empty(())


def save_frames(frames: np.ndarray, shot_name: str, frame_dir: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_path = str(frame_dir / (shot_name + ".mp4"))
    width, height = frames.shape[2], frames.shape[1]
    out = cv2.VideoWriter(frame_path, fourcc, 20, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------------------------- #


def main(cm_dir: Path, et_dir: Path):
    frame_dir = et_dir / "frames"
    frame_dir.mkdir(exist_ok=True, parents=True)

    sample_paths = sorted((et_dir / "bounds").iterdir())

    with PROGRESS:
        task = PROGRESS.add_task(
            "[green]Processing...", total=len(sample_paths), step=0
        )
        for sample_path in sample_paths:
            year_name = sample_path.stem.split("_")[0]
            clip_name = "_".join(sample_path.stem.split("_")[1:-2])

            start_bound, end_bound = np.load(sample_path)

            video_path = cm_dir / "videos" / year_name / (clip_name + ".mkv")
            if not video_path.exists():
                PROGRESS.update(task, advance=1)
                continue

            frames = load_frames(video_path, start_bound, end_bound)
            save_frames(frames, sample_path.stem, frame_dir)

            PROGRESS.update(task, advance=1)


if __name__ == "__main__":
    args = parse_arguments()
    main(**args)
