import argparse
from pathlib import Path
from typing import Tuple

from decord import VideoReader, cpu
from evo.tools.file_interface import read_kitti_poses_file
from matplotlib import colormaps
import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
import rerun as rr
from rerun.components import Material
from scipy.spatial import transform
import torch

# ------------------------------------------------------------------------------------- #


def color_fn(x, cmap="tab10"):
    return colormaps[cmap](x % colormaps[cmap].N)


def log_sample(
    root_name: str,
    traj: np.ndarray,
    K: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    frames: np.ndarray,
    caption: str,
):
    num_cameras = traj.shape[0]

    rr.log(root_name, rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, timeless=True)
    rr.log(
        f"{root_name}/trajectory/points",
        rr.Points3D(traj[:, :3, 3]),
        timeless=True,
    )
    rr.log(
        f"{root_name}/trajectory/line",
        rr.LineStrips3D(
            np.stack((traj[:, :3, 3][:-1], traj[:, :3, 3][1:]), axis=1),
            colors=[(1.0, 0.0, 1.0, 1.0)],
        ),
        timeless=True,
    )

    for k in range(num_cameras):
        rr.set_time_sequence("frame_idx", k)

        frame = (
            frames[k]
            if frames is not None
            else np.zeros((int(K[1, -1] * 2), int(K[0, -1] * 2), 3))
        )
        rr.log(
            f"{root_name}/camera/image",
            rr.Image(frame),
        )
        translation = traj[k][:3, 3]
        rotation_q = transform.Rotation.from_matrix(traj[k][:3, :3]).as_quat()
        rr.log(
            f"{root_name}/camera/image",
            rr.Pinhole(
                image_from_camera=K,
                width=K[0, -1] * 2,
                height=K[1, -1] * 2,
            ),
        )
        rr.log(
            f"{root_name}/camera",
            rr.Transform3D(
                translation=translation,
                rotation=rr.Quaternion(xyzw=rotation_q),
            ),
        )
        rr.set_time_sequence("image", k)

        # Null vertices
        if vertices[k].sum() == 0:
            rr.log(f"{root_name}/human/human", rr.Clear(recursive=False))
            rr.log(f"{root_name}/camera/image/bbox", rr.Clear(recursive=False))
            continue

        rr.log(
            f"{root_name}/human/human",
            rr.Mesh3D(
                vertex_positions=vertices[k],
                indices=faces[k],
                vertex_normals=normals[k],
                mesh_material=Material(albedo_factor=color_fn(0)),
            ),
        )
    rr.log(
        f"{root_name}/caption",
        rr.TextDocument(caption, media_type=rr.MediaType.MARKDOWN),
        timeless=True,
    )


# ------------------------------------------------------------------------------------- #


def parse_arguments() -> Tuple[Path, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Path to the data directory")
    parser.add_argument(
        "--clip-id", "-c", type=str, default=None, help="Optional clip ID"
    )
    parser.add_argument("--random-seed", "-r", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--acceleration",
        "-a",
        type=str,
        choices=["mps", "cuda", "cpu"],
        default="cpu",
        help="Choose the type of acceleration: 'mps' for Apple Silicon, "
        "'cuda' for CUDA acceleration, or None for no acceleration.",
    )
    args = parser.parse_args()
    return (
        args.data_dir,
        args.clip_id,
        args.random_seed,
        args.acceleration,
    )


# ------------------------------------------------------------------------------------- #


def get_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices_pth = torch.from_numpy(vertices)
    faces_pth = torch.from_numpy(faces)

    verts_rgb = torch.ones_like(vertices_pth)
    verts_rgb[:, :, 1] = 0
    textures = TexturesVertex(verts_features=verts_rgb)
    chars = Meshes(verts=vertices_pth, faces=faces_pth, textures=textures)
    normals = chars.verts_normals_padded().numpy()

    return normals, chars


def load_intrinsics(intrinsic_path: str) -> np.ndarray:
    intrinsics = np.load(intrinsic_path)
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def load_frames(video_path: str) -> np.ndarray:
    try:
        with open(video_path, "rb") as f:
            vr = VideoReader(f, ctx=cpu(0))
        frames = vr.get_batch(range(vr.__len__()))
        return frames.asnumpy()
    except:
        return None


# ------------------------------------------------------------------------------------- #


def launch_viz(
    traj_dir: Path,
    frame_dir: Path,
    intrinsic_dir: Path,
    caption_dir: Path,
    vert_dir: Path,
    clip_id: str,
    acceleration: str,
):
    print(f"Visualizing clip {clip_id}")
    traj_path = traj_dir / f"{clip_id}.txt"
    frame_path = frame_dir / f"{clip_id}.mp4"
    intrinsic_path = intrinsic_dir / f"{clip_id}.npy"
    caption_path = caption_dir / f"{clip_id}.txt"
    vert_path = vert_dir / f"{clip_id}.npy"

    raw_traj = np.array(read_kitti_poses_file(traj_path).poses_se3)

    frames = load_frames(frame_path)
    K = load_intrinsics(intrinsic_path)
    caption = str(" ".join(np.loadtxt(caption_path, dtype=str)))
    raw_verts = np.load(vert_path, allow_pickle=True)[()]
    vertices = raw_verts["vertices"]
    faces = raw_verts["faces"]

    if len(faces.shape) == 2:
        num_frames = vertices.shape[0]
        faces = np.repeat(faces[None], [num_frames], axis=0)

    normals, chars = get_normals(vertices, faces)

    rr.init(f"{clip_id}", spawn=True)
    log_sample(
        root_name="world",
        traj=raw_traj,
        K=K,
        vertices=vertices,
        normals=normals,
        faces=faces,
        frames=frames,
        caption=caption,
    )
    input("Press Enter to continue...")


# ------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    data_dir, clip_id, random_seed, acceleration = parse_arguments()

    traj_dir = data_dir / "traj_raw"
    frame_dir = data_dir / "frames"
    intrinsic_dir = data_dir / "intrinsics"
    caption_dir = data_dir / "caption"
    vert_dir = data_dir / "vert_raw"

    np.random.seed(random_seed)
    random_clip = True if clip_id is None else False

    while True:
        if random_clip:
            clip_id = np.random.choice(list(vert_dir.glob("*.npy"))).stem

        launch_viz(
            traj_dir,
            frame_dir,
            intrinsic_dir,
            caption_dir,
            vert_dir,
            clip_id,
            acceleration,
        )
