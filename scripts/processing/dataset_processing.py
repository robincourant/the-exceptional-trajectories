"""
Format trajectory dataset:
    - Crop trajectories greater than a certain length;
    - Gather all chunk information (traj, annotations, character, etc...).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import random

import cv2
import numpy as np
from torchtyping import TensorType
from decord import VideoReader, cpu

from helper.files import load_txt
from helper.progress import PROGRESS

# ------------------------------------------------------------------------------------- #

num_frames, num_vertices, num_faces, width, height = None, None, None, None, None

# ------------------------------------------------------------------------------------- #


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path the dataset root directory",
    )

    parser.add_argument(
        "--do_char",
        "-c",
        action="store_true",
        help="Process char trajectories",
    )
    parser.add_argument(
        "--video_dir",
        "-v",
        type=Path,
        help="Path the video directory",
    )

    parser.add_argument(
        "--max_num_frames",
        "-mf",
        default=300,
        type=int,
        help="Maximum number of frames",
    )

    parser.add_argument(
        "--random-seed",
        "-rs",
        default=42,
        type=int,
        help="Random seed",
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


def load_trajectory(
    shot_path: Path,
) -> Tuple[Dict[str, Any], TensorType["num_frames", 4, 4]]:
    # Skip if trajectory data does not exist / corrupted
    try:
        raw_data = np.load(shot_path, allow_pickle=True)[()]
        raw_trajectory = raw_data["w2c_poses"]
        return raw_data, raw_trajectory
    except:
        return None, None


def load_char(raw_data: Dict[str, Any]) -> Tuple[
    TensorType["num_frames", 3],
    TensorType["num_frames", "num_vertices", 3],
    TensorType["num_faces", 3],
]:
    raw_char = raw_data["char_center"] if "char_center" in raw_data else None
    raw_vertices = raw_data["char_vertices"] if "char_vertices" in raw_data else None
    raw_faces = raw_data["char_faces"] if "char_faces" in raw_data else None
    return raw_char, raw_vertices, raw_faces


# ------------------------------------------------------------------------------------- #


def save_traj(
    trajectory: TensorType["num_frames", 4, 4], shot_name: str, trajectory_dir: Path
):
    chunk_trajectory_path = trajectory_dir / (shot_name + ".txt")
    np.savetxt(
        chunk_trajectory_path,
        trajectory[:, :3].reshape(trajectory.shape[0], -1),
        fmt="%.6e",
    )


def save_intrinsics(intrinsics: TensorType[4], shot_name: str, intrinsics_dir: Path):
    np.save(intrinsics_dir / (shot_name + ".npy"), intrinsics)


def save_char(
    char: TensorType["num_frames", 3],
    raw_vertices: TensorType["num_frames", "num_vertices", 3],
    raw_faces: TensorType["num_faces", 3],
    shot_name: str,
    char_dir: Path,
    vertex_dir: Path,
):
    np.save(char_dir / (shot_name + ".npy"), char)
    np.save(
        vertex_dir / (shot_name + ".npy"),
        {"vertices": raw_vertices, "faces": raw_faces},
    )


def save_frames(frames: np.ndarray, shot_name: str, frame_dir: Path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_path = str(frame_dir / (shot_name + ".mp4"))
    width, height = frames.shape[2], frames.shape[1]
    out = cv2.VideoWriter(frame_path, fourcc, 20, (width, height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------------------------------- #


def generate_dir(
    input_dir: Path,
    do_char: bool,
    video_dir: Path,
    max_num_frames: int,
    random_seed: int,
):
    random.seed(random_seed)

    # Create main directory
    data_dir = input_dir / f"{str(max_num_frames).zfill(4)}"
    # Create subdirectories
    trajectory_dir = data_dir / "traj_raw"
    trajectory_dir.mkdir(exist_ok=True, parents=True)
    intrinsics_dir = data_dir / "intrinsics"
    intrinsics_dir.mkdir(exist_ok=True, parents=True)
    if do_char:
        char_dir = data_dir / "char_raw"
        char_dir.mkdir(exist_ok=True, parents=True)
        vertex_dir = data_dir / "vert_raw"
        vertex_dir.mkdir(exist_ok=True, parents=True)
    do_frames = True if video_dir is not None else False
    if do_frames:
        frame_dir = data_dir / "frames"
        frame_dir.mkdir(exist_ok=True, parents=True)

    # Gather all shot paths to process
    shot_paths = [
        shot_path
        for year_dir in sorted((input_dir / "clean").iterdir())
        for clip_dir in sorted(year_dir.iterdir())
        if clip_dir.is_dir()
        for shot_path in sorted(clip_dir.iterdir())
    ]

    # --------------------------------------------------------------------------------- #
    with PROGRESS:
        task = PROGRESS.add_task("[green]Processing...", total=len(shot_paths), step=0)
        for shot_path in shot_paths:
            year_name, clip_name = shot_path.parts[-3:-1]
            shot_index = shot_path.stem
            shot_name = "_".join([year_name, clip_name, shot_index])

            if do_char and (vertex_dir / (shot_name + ".npy")).exists():
                PROGRESS.update(task, advance=1)
                continue

            # ------------------------------------------------------------------------- #
            # LOAD DATA

            # Load camera trajectory
            raw_data, raw_trajectory = load_trajectory(shot_path)
            if raw_trajectory is None:
                PROGRESS.update(task, advance=1)
                continue

            # Load char trajectory
            if do_char:
                raw_char, raw_vertices, raw_faces = load_char(raw_data)
            else:
                raw_char, raw_vertices, raw_faces = None, None, None
            if do_char and (raw_char is None):
                PROGRESS.update(task, advance=1)
                continue
            if do_char and raw_char.sum() == 0.0:
                PROGRESS.update(task, advance=1)
                continue

            # ------------------------------------------------------------------------- #
            # PROCESS DATA
            num_frames = raw_trajectory.shape[0]
            if num_frames > max_num_frames:
                raw_trajectory = raw_trajectory[:max_num_frames]

                if do_char:
                    raw_char = raw_char[:max_num_frames]
                    if raw_vertices is not None:
                        raw_vertices = raw_vertices[:max_num_frames]

            # Save trajectory
            save_traj(raw_trajectory, shot_name, trajectory_dir)

            # Save intrinsics
            save_intrinsics(raw_data["intrinsics"], shot_name, intrinsics_dir)

            # Save char trajectory
            if do_char:
                save_char(
                    raw_char, raw_vertices, raw_faces, shot_name, char_dir, vertex_dir
                )

            # Load frames
            if do_frames:
                start_bound, end_bound = raw_data["bounds"]
                shot_to_bounds = load_shot(video_dir, year_name, clip_name)
                shot_start_bound = (
                    shot_to_bounds[int(shot_index.split("_")[0])][0]
                    if shot_to_bounds is not None
                    else 0
                )
                start_frame_index = shot_start_bound + start_bound
                end_frame_index = shot_start_bound + min(end_bound, max_num_frames)

                video_path = video_dir / "videos" / year_name / (clip_name + ".mkv")
                if not video_path.exists():
                    PROGRESS.update(task, advance=1)
                    continue

                frames = load_frames(video_path, start_frame_index, end_frame_index)
                save_frames(frames, shot_name, frame_dir)
            # ------------------------------------------------------------------------- #

            PROGRESS.update(task, advance=1)


if __name__ == "__main__":
    args = parse_arguments()
    generate_dir(**args)
