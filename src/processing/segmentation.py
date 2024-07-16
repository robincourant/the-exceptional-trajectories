from itertools import product
from typing import List, Tuple

from evo.core import lie_algebra as lie
import numpy as np
import torch
from scipy.stats import mode
from scipy.spatial.transform import Rotation as R
from torchtyping import TensorType

# ------------------------------------------------------------------------------------- #

num_frames, width, height = None, None, None

# ------------------------------------------------------------------------------------- #

CAM_INDEX_TO_PATTERN = {
    0: "static",
    1: "push_in",
    2: "pull_out",
    3: "boom_bottom",
    6: "boom_top",
    18: "trucking_left",
    9: "trucking_right",
    # ----- #
    12: "trucking_right-boom_bottom",
    15: "trucking_right-boom_top",
    21: "trucking_left-boom_bottom",
    24: "trucking_left-boom_top",
    10: "trucking_right-push_in",
    11: "trucking_right-pull_out",
    19: "trucking_left-push_in",
    20: "trucking_left-pull_out",
    4: "boom_bottom-push_in",
    5: "boom_bottom-pull_out",
    7: "boom_top-push_in",
    8: "boom_top-pull_out",
    # ----- #
    13: "trucking_right-boom_bottom-push_in",
    14: "trucking_right-boom_bottom-pull_out",
    16: "trucking_right-boom_top-push_in",
    17: "trucking_right-boom_top-pull_out",
    22: "trucking_left-boom_bottom-push_in",
    23: "trucking_left-boom_bottom-pull_out",
    25: "trucking_left-boom_top-push_in",
    26: "trucking_left-boom_top-pull_out",
}

BODY_INDEX_TO_PATTERN = {
    0: "static",
    1: "move_forward",
    2: "move_backward",
    3: "move_down",
    6: "move_up",
    18: "move_left",
    9: "move_right",
    # ----- #
    12: "move_right-move_down",
    15: "move_right-move_up",
    21: "move_left-move_down",
    24: "move_left-move_up",
    10: "move_right-move_forward",
    11: "move_right-move_backward",
    19: "move_left-move_forward",
    20: "move_left-move_backward",
    4: "move_down-move_forward",
    5: "move_down-move_backward",
    7: "move_up-move_forward",
    8: "move_up-move_backward",
    # ----- #
    13: "move_right-move_down-move_forward",
    14: "move_right-move_down-move_backward",
    16: "move_right-move_up-move_forward",
    17: "move_right-move_up-move_backward",
    22: "move_left-move_down-move_forward",
    23: "move_left-move_down-move_backward",
    25: "move_left-move_up-move_forward",
    26: "move_left-move_up-move_backward",
}

# ------------------------------------------------------------------------------------- #


def to_euler_angles(
    rotation_mat: TensorType["num_frames", 3, 3]
) -> TensorType["num_frames", 3]:
    rotation_vec = torch.from_numpy(
        np.stack(
            [lie.sst_rotation_from_matrix(r).as_rotvec() for r in rotation_mat.numpy()]
        )
    )
    return rotation_vec


def compute_relative(f_t: TensorType["num_frames", 3]):
    max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 1])]), axis=0)
    xy_f_t = np.divide(
        (abs(f_t[:, 0]) - abs(f_t[:, 1])),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    max_value = np.max(np.stack([abs(f_t[:, 0]), abs(f_t[:, 2])]), axis=0)
    xz_f_t = np.divide(
        abs(f_t[:, 0]) - abs(f_t[:, 2]),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    max_value = np.max(np.stack([abs(f_t[:, 1]), abs(f_t[:, 2])]), axis=0)
    yz_f_t = np.divide(
        abs(f_t[:, 1]) - abs(f_t[:, 2]),
        max_value,
        out=np.zeros_like(max_value),
        where=max_value != 0,
    )
    return xy_f_t, xz_f_t, yz_f_t


def compute_camera_dynamics(w2c_poses: TensorType["num_frames", 4, 4], fps: float):
    w2c_poses_inv = torch.from_numpy(
        np.array([lie.se3_inverse(t) for t in w2c_poses.numpy()])
    )
    velocities = w2c_poses_inv[:-1].to(float) @ w2c_poses[1:].to(float)

    # --------------------------------------------------------------------------------- #
    # Translation velocity
    t_velocities = fps * velocities[:, :3, 3]
    t_xy_velocity, t_xz_velocity, t_yz_velocity = compute_relative(t_velocities)
    t_vels = (t_velocities, t_xy_velocity, t_xz_velocity, t_yz_velocity)
    # --------------------------------------------------------------------------------- #
    # Rotation velocity
    a_velocities = to_euler_angles(velocities[:, :3, :3])
    a_xy_velocity, a_xz_velocity, a_yz_velocity = compute_relative(a_velocities)
    a_vels = (a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity)

    return velocities, t_vels, a_vels


def compute_char_dynamics(char_points: TensorType["num_frames", 3], fps: float):
    velocities = fps * (char_points[1:] - char_points[:-1])
    xy_velocity, xz_velocity, yz_velocity = compute_relative(velocities)
    return velocities, xy_velocity, xz_velocity, yz_velocity


def normalize_vectors(
    input_vectors: TensorType["num_frames", 3]
) -> TensorType["num_frames", 3]:
    norm_input_vectors = torch.linalg.norm(input_vectors.clone(), dim=1)[:, None]
    normed_vectors = input_vectors.clone() / norm_input_vectors
    return normed_vectors


def get_vector_angles(vec1, vec2):
    normed_vec1 = normalize_vectors(vec1)
    normed_vec2 = normalize_vectors(vec2)

    R_axis = torch.cross(normed_vec1, normed_vec2)
    R_axis_norm = torch.linalg.norm(R_axis, dim=1)
    R_angle = torch.sum(normed_vec1 * normed_vec2, dim=1)

    K = torch.zeros((R_axis.shape[0], 3, 3))
    K[:, 0, 1] = -R_axis[:, 2]
    K[:, 0, 2] = R_axis[:, 1]
    K[:, 1, 0] = R_axis[:, 2]
    K[:, 1, 2] = -R_axis[:, 0]
    K[:, 2, 0] = -R_axis[:, 1]
    K[:, 2, 1] = R_axis[:, 0]

    R_matrix = torch.eye(3).repeat(R_axis.shape[0], 1, 1)
    R_matrix += K
    R_matrix += ((1 - R_angle) / R_axis_norm**2)[:, None, None] * (K @ K)

    vector_angles = R.from_matrix(R_matrix.numpy()).as_euler("yxz", degrees=True)

    return vector_angles


def build_poses(
    t: TensorType["num_frames", 3], direction: TensorType["num_frames", 3]
) -> TensorType["num_frames", 4, 4]:
    # Step 1: Normalize the direction vector
    forward = direction / torch.linalg.norm(direction, dim=1)[:, None]

    # Step 2: Compute up and right vectors
    up = torch.tensor([[0, 1.0, 0]]).repeat(forward.shape[0], 1)  # Arbitrary up vector
    right = torch.cross(forward, up)
    up = torch.cross(right, forward)

    # Step 3: Form the rotation matrix
    R = torch.stack([right, -up, forward], dim=1).mT

    # Step 5: Construct the SE(3) matrix
    poses = torch.eye(4).repeat(t.shape[0], 1, 1)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = t

    return poses


# ------------------------------------------------------------------------------------- #


def perform_segmentation(
    velocities: TensorType["num_frames-1", 3],
    xy_velocity: TensorType["num_frames-1", 3],
    xz_velocity: TensorType["num_frames-1", 3],
    yz_velocity: TensorType["num_frames-1", 3],
    static_threshold: float,
    diff_threshold: float,
) -> List[int]:
    segments = torch.zeros(velocities.shape[0])
    segment_patterns = [torch.tensor(x) for x in product([0, 1, -1], repeat=3)]
    pattern_to_index = {
        tuple(pattern.numpy()): index for index, pattern in enumerate(segment_patterns)
    }

    for sample_index, sample_velocity in enumerate(velocities):
        sample_pattern = abs(sample_velocity) > static_threshold

        # XY
        if (sample_pattern == torch.tensor([1, 1, 0])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])

        # XZ
        elif (sample_pattern == torch.tensor([1, 0, 1])).all():
            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([1, 0, 0])
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # YZ
        elif (sample_pattern == torch.tensor([0, 1, 1])).all():
            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern = torch.tensor([0, 1, 0])
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern = torch.tensor([0, 0, 1])

        # XYZ
        elif (sample_pattern == torch.tensor([1, 1, 1])).all():
            if xy_velocity[sample_index] > diff_threshold:
                sample_pattern[1] = 0
            elif xy_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if xz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif xz_velocity[sample_index] < -diff_threshold:
                sample_pattern[0] = 0

            if yz_velocity[sample_index] > diff_threshold:
                sample_pattern[2] = 0
            elif yz_velocity[sample_index] < -diff_threshold:
                sample_pattern[1] = 0

        sample_pattern = torch.sign(sample_velocity) * sample_pattern
        segments[sample_index] = pattern_to_index[tuple(sample_pattern.numpy())]

    return np.array(segments, dtype=int)


# ------------------------------------------------------------------------------------- #


def smooth_segments(arr: List[int], window_size: int) -> List[int]:
    smoothed_arr = arr.copy()

    if len(arr) < window_size:
        return smoothed_arr

    half_window = window_size // 2
    # Handle the first half_window elements
    for i in range(half_window):
        window = arr[: i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    for i in range(half_window, len(arr) - half_window):
        window = arr[i - half_window : i + half_window + 1]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    # Handle the last half_window elements
    for i in range(len(arr) - half_window, len(arr)):
        window = arr[i - half_window :]
        most_frequent = mode(window, keepdims=False).mode
        smoothed_arr[i] = most_frequent

    return smoothed_arr


def remove_short_chunks(arr: List[int], min_chunk_size: int) -> List[int]:
    def remove_chunk(chunks):
        if len(chunks) == 1:
            return False, chunks

        chunk_lenghts = [(end - start) + 1 for _, start, end in chunks]
        chunk_index = np.argmin(chunk_lenghts)
        chunk_length = chunk_lenghts[chunk_index]
        if chunk_length < min_chunk_size:
            _, start, end = chunks[chunk_index]

            # Check if the chunk is at the beginning
            if chunk_index == 0:
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index + 1] = (segment_r, start_r - chunk_length, end_r)

            elif chunk_index == len(chunks) - 1:
                segment_l, start_l, end_l = chunks[chunk_index - 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + chunk_length)

            else:
                if chunk_length % 2 == 0:
                    half_length_l = chunk_length // 2
                    half_length_r = chunk_length // 2
                else:
                    half_length_l = (chunk_length // 2) + 1
                    half_length_r = chunk_length // 2

                segment_l, start_l, end_l = chunks[chunk_index - 1]
                segment_r, start_r, end_r = chunks[chunk_index + 1]
                chunks[chunk_index - 1] = (segment_l, start_l, end_l + half_length_l)
                chunks[chunk_index + 1] = (segment_r, start_r - half_length_r, end_r)

            chunks.pop(chunk_index)

        return chunk_length < min_chunk_size, chunks

    chunks = find_consecutive_chunks(arr)
    keep_removing, chunks = remove_chunk(chunks)
    while keep_removing:
        keep_removing, chunks = remove_chunk(chunks)

    merged_chunks = []
    for segment, start, end in chunks:
        merged_chunks.extend([segment] * ((end - start) + 1))

    return merged_chunks


def find_consecutive_chunks(arr: List[int]) -> List[Tuple[int, int, int]]:
    chunks = []
    start_index = 0
    for i in range(1, len(arr)):
        if arr[i] != arr[i - 1]:
            end_index = i - 1
            if end_index >= start_index:
                chunks.append((arr[start_index], start_index, end_index))
            start_index = i

    # Add the last chunk if the array ends with consecutive similar digits
    if start_index < len(arr):
        chunks.append((arr[start_index], start_index, len(arr) - 1))

    return chunks


# ------------------------------------------------------------------------------------- #


def segment_rigidbody_trajectories(
    w2c_poses: TensorType["num_frames", 4, 4],
    fps: float,
    cam_static_threshold: float,
    cam_diff_threshold: float,
    smoothing_window_size: int,
    min_chunk_size: int,
) -> List[int]:
    velocities, t_vels, a_vels = compute_camera_dynamics(w2c_poses, fps=fps)
    cam_velocities, cam_xy_velocity, cam_xz_velocity, cam_yz_velocity = t_vels
    # a_velocities, a_xy_velocity, a_xz_velocity, a_yz_velocity = a_vels
    # ----------------------------------------------------------------------------- #

    # Translation segments
    cam_segments = perform_segmentation(
        cam_velocities,
        cam_xy_velocity,
        cam_xz_velocity,
        cam_yz_velocity,
        static_threshold=cam_static_threshold,
        diff_threshold=cam_diff_threshold,
    )
    cam_segments = smooth_segments(cam_segments, smoothing_window_size)
    cam_segments = remove_short_chunks(cam_segments, min_chunk_size)

    return cam_segments


def segment_translation_trajectories(
    char_center: TensorType["num_frames", 3],
    fps: float,
    char_static_threshold: float,
    char_diff_threshold: float,
    smoothing_window_size: int,
    min_chunk_size: int,
) -> List[int]:
    center_vels = compute_char_dynamics(char_center, fps=fps)
    c_velocities, c_xy_velocity, c_xz_velocity, c_yz_velocity = center_vels
    # ----------------------------------------------------------------------------- #

    c_segments = perform_segmentation(
        c_velocities,
        c_xy_velocity,
        c_xz_velocity,
        c_yz_velocity,
        static_threshold=char_static_threshold,
        diff_threshold=char_diff_threshold,
    )
    c_segments = smooth_segments(c_segments, smoothing_window_size)
    c_segments = remove_short_chunks(c_segments, min_chunk_size)

    return c_segments
