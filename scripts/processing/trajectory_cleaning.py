"""Clean all raw trajectories."""

import argparse
from pathlib import Path

import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import torch

from helper.camera import invert_cams
from helper.progress import PROGRESS
from src.processing.cleaning import clean_trajectories

# ------------------------------------------------------------------------------------- #

FRONT_VERT_INDEX = 3146  # Facing direction (331 front nose, 3146 front center hips)

# ------------------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--overwrite", "-o", action="store_true")
    return parser.parse_args()


# ------------------------------------------------------------------------------------- #


def get_char_orientations(vertices, faces):
    textures = TexturesVertex(verts_features=torch.ones_like(vertices))
    chars = Meshes(
        verts=vertices,
        faces=faces[None].repeat(vertices.shape[0], 1, 1),
        textures=textures,
    )
    char_orientations = chars.verts_normals_padded()[:, FRONT_VERT_INDEX]
    del chars, textures
    return char_orientations


# ------------------------------------------------------------------------------------- #


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    input_dir = data_dir / "raw"
    output_dir = data_dir / "clean"
    overwrite = args.overwrite

    raw_traj_paths = [
        shot_dir
        for year_dir in (input_dir / "traj").iterdir()
        if year_dir.is_dir()
        for clip_dir in sorted(year_dir.iterdir())
        if clip_dir.is_dir()
        for shot_dir in sorted(clip_dir.iterdir())
    ]
    char_dir = input_dir / "char"

    with PROGRESS:
        task = PROGRESS.add_task(
            "[green]Processing...", total=len(raw_traj_paths), step=0
        )
        for _, raw_path in enumerate(raw_traj_paths):
            year, videoid, shot_idx = raw_path.parts[-3:]
            chunk_dir = output_dir / year / videoid

            if (
                (not overwrite)
                and chunk_dir.exists()
                and bool(len(list(chunk_dir.glob(f"{shot_idx[:-4]}*"))))
            ):
                PROGRESS.update(task, advance=1)
                continue

            try:
                raw_data = np.load(raw_path, allow_pickle=True)[()]
            except:
                PROGRESS.log(f"No data found at {raw_path}")
                PROGRESS.update(task, advance=1)
                continue

            intrinsics = raw_data["intrinsics"][0]
            raw_c2w_poses = raw_data["pose_se3"].to(torch.float32)
            raw_w2c_poses = invert_cams(raw_c2w_poses)
            del raw_data

            char_path = char_dir / year / videoid / (shot_idx[:-3] + "npy")
            raw_char = np.load(char_path, allow_pickle=True)[()]

            if len(raw_char["main_characters"]) > 0:
                main_char_index = raw_char["main_characters"][0]
                raw_verts = raw_char["vertices_tracks"][main_char_index].to(
                    torch.float32
                )
                char_orientations = None
            else:
                raw_verts, char_orientations = None, None

            clean_chunks = clean_trajectories(
                raw_w2c_poses, raw_verts, char_orientations
            )
            if raw_verts is not None:
                (
                    clean_chunks,
                    clean_verts,
                    clean_fronts,
                    clean_char,
                ) = clean_chunks

            for chunk_index, (chunk_bounds, chunk_w2c) in enumerate(clean_chunks):
                clean_data = dict(
                    bounds=chunk_bounds,
                    w2c_poses=chunk_w2c.cpu().numpy(),
                    intrinsics=intrinsics.cpu().numpy(),
                )
                if raw_verts is not None:
                    (start, end), chunk_verts = clean_verts[chunk_index]
                    clean_data["char_center"] = chunk_verts.cpu().numpy()

                    (start, end), chunk_fronts = clean_fronts[chunk_index]
                    clean_data["char_fronts"] = chunk_fronts.cpu().numpy()

                    (start, end), chunk_chars = clean_char[chunk_index]
                    clean_data["char_vertices"] = chunk_chars.cpu().numpy()

                    clean_data["char_faces"] = raw_char["faces"].cpu().numpy()

                chunk_path = chunk_dir / (
                    shot_idx[:-4] + "_" + str(chunk_index).zfill(5) + ".npy"
                )
                chunk_path.parent.mkdir(parents=True, exist_ok=True)

                np.save(chunk_path, clean_data)
                # Delete unnecessary variables
                del clean_data

            # Delete unnecessary variables
            del raw_verts, char_orientations
            PROGRESS.update(task, advance=1)
