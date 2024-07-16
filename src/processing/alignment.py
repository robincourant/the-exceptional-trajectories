from collections import defaultdict
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from rich.console import Console
from scipy.ndimage import binary_fill_holes, binary_dilation
import torch
import torch.nn.functional as F
from torchmin import minimize
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #


CONSOLE = Console(width=170)

height, width = None, None
num_tracks, num_frames, num_samples, num_verts, num_faces = None, None, None, None, None

Verts = TensorType["num_tracks", "num_frames", "num_verts", 3]
Faces = TensorType["num_faces", 3]

NUM_VERTICES = 6890
NUM_FACES = 13776


# ------------------------------------------------------------------------------------- #


def get_char_coverages(
    vertices: Verts,
    faces: Faces,
    w2c_poses: TensorType["num_frames", 4, 4],
    c2w_poses: TensorType["num_frames", 4, 4],
    intrinsics: TensorType["num_frames", 4],
) -> TensorType["num_tracks", "num_frames"]:
    """
    Calculates the 2d char coverages on screen for each track at each frame.

    :param vertices: 3D vertices of the char for each track at each frame.
    :param faces: faces of the char.
    :param w2c_poses: world-to-camera poses for each frame.
    :param c2w_poses: camera-to-world poses for each frame.
    :param intrinsics: camera intrinsics for each frame.

    :return: array containing the char coverages for each track at each frame.
    """
    fx, fy, cx, cy = intrinsics[0]
    height, width = int(cy * 2), int(cx * 2)
    K = torch.eye(3)
    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = fx, fy, cx, cy
    frame_area = height * width

    num_tracks, num_frames = vertices.shape[:2]
    char_coverages = torch.zeros((num_tracks, num_frames))
    for k in range(num_frames):
        Rt = torch.eye((4))
        Rt[:3, :3] = w2c_poses[k][:3, :3]
        Rt[:3, 3] = w2c_poses[k][:3, 3]
        M = K @ w2c_poses[k][:3, :3]
        m = np.array(M[2]) / np.linalg.norm(M[2])
        for i in range(num_tracks):
            d = (vertices[i, k] - c2w_poses[k, :3, 3].numpy()) @ m
            # Mesh behind the camera
            if d.max() < 1:
                char_coverages[i, k] = 0.0
                continue

            verts_h = np.hstack((vertices[i, k], np.ones((vertices[i, k].shape[0], 1))))
            char_mask = project_char(verts_h, faces, Rt, K, height, width)
            x1, y1, x2, y2 = find_bounding_box(char_mask)
            bbox_area = ((x2 - x1) * (y2 - y1)) / frame_area
            char_coverages[i, k] = bbox_area if bbox_area > 0.01 else 0.0

    return char_coverages


def project_char(
    verts_h: TensorType["num_verts", 4],
    faces: Faces,
    Rt: TensorType[4, 4],
    K: TensorType[3, 3],
    height: int,
    width: int,
) -> TensorType["height", "width"]:
    """
    Projects the 3D char vertices onto the 2D image plane.

    :param verts_h: 3D vertices of the char with homogeneous coordinates.
    :param faces: faces of the char.
    :param Rt: 3x3 rotation and 3x1 translation vector from world to camera coords.
    :param K: camera intrinsic matrix.
    :param height: height of the image.
    :param width: width of the image.
    :return: binary mask representing the projected char on the image plane.
    """
    face_centers = np.mean(verts_h[faces], axis=1)
    coords = np.concatenate((verts_h, face_centers), axis=0)
    coords = verts_h
    Pc = np.matmul(Rt[:3], coords.T)
    p = np.matmul(K, Pc)
    coords_2d = (p / p[2])[:-1].T
    coords_2d = coords_2d.numpy().astype(int)
    mask_indices = coords_2d[
        (coords_2d[:, 0] > -1)
        & (coords_2d[:, 1] > -1)
        & (coords_2d[:, 0] < width)
        & (coords_2d[:, 1] < height)
    ]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[mask_indices[:, 1], mask_indices[:, 0]] = 1
    mask = mask.astype(np.uint8)
    mask = binary_dilation(mask > 0).astype(np.uint8)
    mask = binary_dilation(mask > 0).astype(np.uint8)
    mask = torch.from_numpy(binary_fill_holes(mask > 0).astype(np.uint8))

    return mask


def find_bounding_box(mask: TensorType["height", "width"]) -> Tuple[int, int, int, int]:
    """
    Finds the bounding box of the non-zero elements in the mask.

    :param mask: binary mask representing the projected char on the image plane.
    :return: bounding-box coordinates of the mask (left, bot, right, top).
    """
    # Find non-zero indices along each axis
    non_zero_rows = torch.nonzero(torch.any(mask, dim=1))
    non_zero_cols = torch.nonzero(torch.any(mask, dim=0))

    # Find minimum and maximum values along each axis
    min_row = min(non_zero_rows) if len(non_zero_rows) else torch.tensor(0.0)
    max_row = max(non_zero_rows) if len(non_zero_rows) else torch.tensor(0.0)
    min_col = min(non_zero_cols) if len(non_zero_cols) else torch.tensor(0.0)
    max_col = max(non_zero_cols) if len(non_zero_cols) else torch.tensor(0.0)

    # Create bounding box
    bounding_box = np.array(
        [min_col.item(), min_row.item(), max_col.item(), max_row.item()]
    )
    return bounding_box


def transform_pose(
    pose: TensorType[4, 4], scale: float, transform: TensorType[3]
) -> TensorType[4, 4]:
    """Transforms the given pose by scaling and translating it."""
    new_pose = pose.clone()
    new_pose[:3, 3] *= scale
    new_pose[:3, 3] += transform
    return new_pose


def transform_verts(
    verts: Verts, raw_pose: TensorType[4, 4], aligned_pose: TensorType[4, 4]
) -> Verts:
    """
    Transforms the vertices by applying the inverse of the aligned pose and the raw pose.
    """
    new_verts = torch.cat(
        (verts.clone(), torch.ones((verts.shape[0], verts.shape[1], 1))), dim=-1
    )
    if new_verts.shape[0] != 0:
        delta_p = aligned_pose.inverse() @ raw_pose
        new_verts = (delta_p[None, None] @ new_verts[..., None]).squeeze(-1)

    return new_verts[..., :3] / new_verts[..., 3, None]


def least_square(
    traj_1: TensorType["num_samples", 4, 4], traj_2: TensorType["num_samples", 4, 4]
) -> Tuple[bool, TensorType[4]]:
    """
    This function performs least squares optimization to align two trajectories.

        | t1_target 1 0 0 | | s  | = | t1_ref |
        | t2_target 0 1 0 | | T1 |   | t2_ref |
        | t3_target 0 0 1 | | T2 |   | t3_ref |
                            | T3 |

    :param traj_1: reference trajectory.
    :param traj_2: target trajectory.

    :return: optimization success boolean and the optimized params [s, T1, T2, T3].
    """

    def objective_function(params, x_data, y_data):
        predicted_values = x_data @ params
        residuals = torch.linalg.norm((y_data - predicted_values), ord=2)
        return residuals

    # Get the translations
    t1 = traj_1[:, :3, 3]
    t2 = traj_2[:, :3, 3]
    num_samples = t1.shape[0]
    flat_id = torch.eye(3)[None].repeat((num_samples, 1, 1)).reshape(num_samples, -1)
    T2 = torch.hstack([t2, flat_id]).reshape(num_samples, 4, 3).mT

    # Compute the mean-centered trajectories
    t1_m, t2_m = t1 - t1.mean(axis=0), t2 - t2.mean(axis=0)
    if torch.isnan(t1_m.mean()) or torch.isinf(t1_m.mean()):
        t1_m = torch.zeros_like(t1_m)
    if torch.isinf(t2_m.mean()) or torch.isinf(t2_m.mean()):
        t2_m = torch.zeros_like(t2_m)

    # Compute the initial guess for the scale and translation parameters
    s = torch.nan_to_num((t1_m / t2_m).mean()[None])
    ta = (t1_m - t2_m).mean(axis=0)
    initial_guess = torch.cat((s, ta))

    # Define the objective function with the data
    S = partial(objective_function, x_data=T2, y_data=t1)

    # Perform the optimization
    res = minimize(S, initial_guess, method="bfgs")

    return res.success, res.x


def gather_data(
    data: Dict[Tuple[int, int], Dict[str, TensorType]]
) -> Dict[int, List[Tuple[TensorType[4, 4], TensorType[4], Verts, Faces]]]:
    index_to_cam = defaultdict(list)
    for start_index, end_index in sorted(data):
        poses = data[(start_index, end_index)]["pose_se3"]
        intrinsics = data[(start_index, end_index)]["intrinsics"]

        # Check if the data contains simpl_vertices and simpl_faces
        if "vertices" in data[(start_index, end_index)]:
            vertices = data[(start_index, end_index)]["vertices"]
            faces = data[(start_index, end_index)]["faces"]

        # NOTE: to remove (simpl_vertices, simpl_faces) --> (vertices, faces)
        elif "simpl_vertices" in data[(start_index, end_index)]:
            CONSOLE.print("[bold][red]WARNING: using simpl_{vertices/simpl_faces}")
            vertices = data[(start_index, end_index)]["simpl_vertices"]
            faces = data[(start_index, end_index)]["simpl_faces"]

        # If not, create empty tensors for vertices and faces
        else:
            # CONSOLE.print("[bold][red]WARNING: empty char")
            vertices = torch.empty((0, poses.shape[0], NUM_VERTICES, 3))
            faces = torch.empty((NUM_FACES, 3))

        for camera_index in range(end_index - start_index):
            index_to_cam[start_index + camera_index].append(
                (
                    poses[camera_index],
                    intrinsics[camera_index],
                    vertices[:, camera_index],
                    faces,
                )
            )

    return dict(index_to_cam)


def align_data(
    index_to_cam: Dict[int, List[Tuple[TensorType[4, 4], TensorType[4], Verts, Faces]]],
    alignment: bool,
) -> Dict[str, TensorType]:
    aligned_data = defaultdict(list)
    scale, transform, initial_scale = 1, torch.zeros((3,)), True
    traj_1, traj_2, raw_poses = [], [], []
    for camera_index in sorted(index_to_cam):
        # Check if alignment is required
        if alignment:
            # Check if there are overlapping cameras
            if len(index_to_cam[camera_index]) > 1:
                pose_1, _, verts_1, _ = index_to_cam[camera_index][0]
                traj_1.append(transform_pose(pose_1, scale, transform))
                pose_2, _, verts_2, _ = index_to_cam[camera_index][1]
                traj_2.append(pose_2)
            # If there are not, align the buffered overlapping trajectories
            elif len(traj_2) > 0:
                success, res = least_square(torch.stack(traj_1), torch.stack(traj_2))
                scale, transform = res[0], res[1:]
                traj_1, traj_2 = [], []

            # Update initial (not aligned poses and vertices) if the 1st updated scale<0
            if initial_scale and not isinstance(scale, int):
                initial_scale = False
                if scale < 0:
                    fixed_poses, fixed_verts = [], []
                    for _pose, _raw_pose, _verts in zip(
                        aligned_data["pose_se3"], raw_poses, aligned_data["vertices"]
                    ):
                        _pose[:3, 3] *= -1
                        fixed_poses.append(_pose)
                        fixed_verts.append(transform_verts(_verts, _raw_pose, _pose))
                    aligned_data["pose_se3"] = fixed_poses
                    aligned_data["vertices"] = fixed_verts

        # Apply computed transforms
        raw_pose, intr, raw_verts, faces = index_to_cam[camera_index][0]
        aligned_pose = transform_pose(raw_pose, scale, transform)
        if scale < 0:
            aligned_pose[:3, 3] *= -1
        aligned_verts = transform_verts(raw_verts, raw_pose, aligned_pose)

        # Compute normals based on aligned aligned_verts
        faces = faces.expand(aligned_verts.shape[0], faces.shape[-2], 3)
        textures = TexturesVertex(verts_features=torch.ones_like(aligned_verts))
        chars = Meshes(verts=aligned_verts, faces=faces, textures=textures)
        aligned_norms = chars.verts_normals_padded()

        # Store the aligned data
        raw_poses.append(raw_pose)
        aligned_data["pose_se3"].append(aligned_pose)
        aligned_data["intrinsics"].append(intr)
        aligned_data["vertices"].append(aligned_verts)
        aligned_data["normals"].append(aligned_norms)

        if faces.shape[0] != 0 and len(aligned_data["faces"]) == 0:
            aligned_data["faces"] = faces

    return dict(aligned_data)


# ------------------------------------------------------------------------------------- #


def process_data(
    data: Dict[Tuple[int, int], Dict[str, TensorType]], alignment: bool = True
) -> Dict[str, TensorType]:
    """
    This function processes the data by aligning the vertices and poses.
    It also pads the vertices to have the same number of tracks.

    :param data: a dictionary containing the start and end indices as keys and another
        dictionary as values. The inner dictionary contains the pose, intrinsics,
        vertices, and faces.
    :param alignment: a boolean indicating whether to align the data or not.

    :return: a dictionary containing the aligned pose, intrinsics, vertices, and faces.
    """

    def pad_tracks(raw_in: List[TensorType], max_num_tracks: int) -> List[TensorType]:
        padded_out = []
        for x in raw_in:
            if x.shape[1] == 0:
                x = torch.empty((max_num_tracks - x.shape[0], NUM_VERTICES, 3))
            x_paddded = F.pad(x, (0, 0, 0, 0, max_num_tracks - x.shape[0], 0))
            padded_out.append(x_paddded)

        return padded_out

    index_to_cam = gather_data(data)
    aligned_data = align_data(index_to_cam, alignment)

    # Process the aligned data (pad vertices and normals, stack tensors)
    max_num_tracks = max([v.shape[0] for v in aligned_data["vertices"]])
    padded_vertices = pad_tracks(aligned_data["vertices"], max_num_tracks)
    padded_normals = pad_tracks(aligned_data["normals"], max_num_tracks)

    aligned_data["pose_se3"] = torch.stack(aligned_data["pose_se3"])
    aligned_data["intrinsics"] = torch.stack(aligned_data["intrinsics"])
    aligned_data["vertices"] = torch.stack(padded_vertices).permute(1, 0, 2, 3)
    aligned_data["normals"] = torch.stack(padded_normals).permute(1, 0, 2, 3)
    if len(aligned_data["vertices"]) != 0:
        aligned_data["faces"] = aligned_data["faces"][0]
    else:
        aligned_data["faces"] = torch.empty((0, NUM_FACES, 3))

    return aligned_data
