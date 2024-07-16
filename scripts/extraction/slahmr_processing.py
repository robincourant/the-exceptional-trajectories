"""Merge all SLAHMR of a shot into one file."""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm
import torch

from lib.slahmr.slahmr.optim.output import get_results_paths, load_result
from lib.slahmr.slahmr.geometry import camera as cam_util
from lib.slahmr.slahmr.eval.tools import load_results_all

# ------------------------------------------------------------------------------------- #


def parse_args():
    """
    Parse command line arguments.

    :returns: The parsed arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--dirname", "-d", type=str, default="smooth_fit")
    parser.add_argument("--overwrite", "-o", action="store_true")
    return parser.parse_args()


def merge_results(
    chunk_dir: Path, dirname: str
) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """
    Merge the results of a chunk.

    :param chunk_dir: The chunk directory.
    :type chunk_dir: Path
    :param dirname: The directory name.
    :type dirname: str
    :returns: The merged cameras, start index, and end index.
    :rtype: Tuple[Dict[str, torch.Tensor], int, int]
    """
    sub_cameras = dict()
    res_path_dict = get_results_paths(chunk_dir / dirname)
    if not res_path_dict:
        return sub_cameras, -1, -1
    it = sorted(res_path_dict.keys())[-1]
    res_cam = load_result(res_path_dict[it])["world"]
    res_simpl = load_results_all(chunk_dir / "smooth_fit", "cuda")

    pose = cam_util.make_4x4_pose(res_cam["cam_R"][0], res_cam["cam_t"][0])
    num_cams = pose.shape[0]

    sub_cameras["pose_se3"] = pose
    sub_cameras["intrinsics"] = res_cam["intrins"][None].repeat((num_cams, 1))
    if res_simpl:
        # NOTE: simpl_joints --> joints
        sub_cameras["joints"] = res_simpl["joints"]
        sub_cameras["vertices"] = res_simpl["vertices"]
        sub_cameras["faces"] = res_simpl["faces"]
    start_index, end_index = chunk_dir.name.split("_")[-2:]
    end_index = min(int(end_index), int(start_index) + num_cams)

    return sub_cameras, int(start_index), int(end_index)


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir
    dirname = args.dirname
    overwrite = args.overwrite

    for year_dir in tqdm(sorted(input_dir.iterdir())):
        if not year_dir.is_dir():
            continue
        for clip_dir in tqdm(sorted(year_dir.iterdir())):
            if not clip_dir.is_dir():
                continue

            # Merge all cameras of a shot into one file
            for shot_dir in sorted(clip_dir.iterdir()):
                # Remove existing merged file if it exists
                if overwrite:
                    (shot_dir / f"all_{dirname}.npy").unlink(missing_ok=True)
                # Pass exisiting merged file
                else:
                    if (shot_dir / f"all_{dirname}.npy").exists():
                        continue

                slahmr_out_dir = shot_dir / "slahmr_out"
                if not shot_dir.is_dir() or not slahmr_out_dir.exists():
                    continue

                shot_cameras = dict()
                for chunk_dir in sorted(slahmr_out_dir.iterdir()):
                    if not chunk_dir.is_dir():
                        continue
                    cameras, start_index, end_index = merge_results(chunk_dir, dirname)
                    if start_index == -1:
                        continue
                    shot_cameras[start_index, end_index] = cameras

                if not shot_cameras:
                    continue

                np.save(shot_dir / f"all_{dirname}.npy", shot_cameras)
