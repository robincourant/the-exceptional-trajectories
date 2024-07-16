import argparse
from copy import deepcopy
from pathlib import Path

from evo.tools.file_interface import read_kitti_poses_file
import numpy as np
from tqdm import tqdm


def parse_arguments():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path the dataset root directory",
    )
    args = parser.parse_args()

    return args.data_dir


if __name__ == "__main__":
    data_dir = parse_arguments()

    traj_dir = data_dir / "traj_raw"
    char_dir = data_dir / "char_raw"
    vert_dir = data_dir / "vert_raw"
    traj_out_dir = data_dir / "traj"
    char_out_dir = data_dir / "char"
    vert_out_dir = data_dir / "vert"
    traj_out_dir.mkdir(exist_ok=True, parents=True)
    char_out_dir.mkdir(exist_ok=True, parents=True)
    vert_out_dir.mkdir(exist_ok=True, parents=True)

    for traj_filename in tqdm(list((data_dir / "traj_raw").iterdir())):
        filename = traj_filename.stem

        raw_traj = read_kitti_poses_file(traj_filename)
        raw_char = np.load(char_dir / f"{filename}.npy")

        try:
            raw_verts = np.load(vert_dir / f"{filename}.npy", allow_pickle=True)[()]
        except:
            continue

        # Take origin as the first char point
        origin = deepcopy(raw_char[0])

        # Shift char
        shifted_char = raw_char - origin

        # Shift trajectory
        shifted_traj = np.stack(raw_traj.poses_se3)
        shifted_traj[:, :3, 3] -= origin

        # Shift vertices
        shifted_verts = deepcopy(raw_verts)
        if shifted_verts["vertices"] is not None:
            shifted_verts["vertices"] -= origin

        np.save(char_out_dir / f"{filename}.npy", shifted_char)
        np.savetxt(
            traj_out_dir / f"{filename}.txt",
            shifted_traj[:, :3].reshape(shifted_traj.shape[0], -1),
            fmt="%.6e",
        )
        np.save(vert_out_dir / f"{filename}.npy", shifted_verts)
