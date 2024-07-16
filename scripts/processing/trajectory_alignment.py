"""Align trajectories from `all_{}.npy` files."""

import argparse
from pathlib import Path

import numpy as np
import torch

from helper.progress import PROGRESS
from src.processing.alignment import process_data
from src.processing.tracker import track_bodies


# ------------------------------------------------------------------------------------- #

MAX_TRAJ_LENGTH = 1000  # ie: 40 seconds at 25 fps

# ------------------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dirname", "-d", type=str, default="smooth_fit")
    parser.add_argument("--overwrite", "-o", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir / "raw"
    dirname = args.dirname
    overwrite = args.overwrite

    shot_paths = [
        shot_dir / f"all_{dirname}.npy"
        for year_dir in sorted([x for x in input_dir.iterdir() if x.is_dir()])
        for clip_dir in sorted(year_dir.iterdir())
        if clip_dir.is_dir()
        for shot_dir in sorted(clip_dir.iterdir())
        if (shot_dir / f"all_{dirname}.npy").exists()
    ]

    with PROGRESS:
        task = PROGRESS.add_task("[green]Processing...", total=len(shot_paths), step=0)
        for raw_trajectory_path in shot_paths:
            year, videoid, shot_idx = raw_trajectory_path.parts[-4:-1]

            char_path = output_dir / "char" / year / videoid / (shot_idx + ".npy")
            traj_path = output_dir / "traj" / year / videoid / (shot_idx + ".npy")

            if (not overwrite) and traj_path.exists() and (char_path.exists()):
                PROGRESS.update(task, advance=1)
                continue

            raw_trajectory = np.load(raw_trajectory_path, allow_pickle=True)[()]
            if list(raw_trajectory.keys())[-1][1] > MAX_TRAJ_LENGTH:
                PROGRESS.update(task, advance=1)
                continue

            try:
                aligned_trajectory = process_data(raw_trajectory)
            except torch._C._LinAlgError:
                PROGRESS.log(f"Failed to align {raw_trajectory_path}.")
                PROGRESS.update(task, advance=1)
                continue
            tracked_trajectory = track_bodies(aligned_trajectory)

            char_data, traj_data = {}, {}

            traj_data["pose_se3"] = tracked_trajectory["pose_se3"].clone().half()
            traj_data["intrinsics"] = tracked_trajectory["intrinsics"].clone().half()

            char_data["vertices"] = tracked_trajectory["vertices"].clone().half()
            char_data["faces"] = tracked_trajectory["faces"].clone().to(torch.int16)
            char_data["vertices_tracks"] = (
                tracked_trajectory["vertices_tracks"].clone().half()
            )
            char_data["main_characters"] = tracked_trajectory["main_characters"]

            char_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(char_path, char_data)
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(traj_path, traj_data)

            PROGRESS.update(task, advance=1)
