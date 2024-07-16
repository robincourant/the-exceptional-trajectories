"""
https://github.com/abewley/sort/blob/master/sort.py
https://github.com/xinshuoweng/AB3DMOT/blob/master/AB3DMOT_libs/model.py
"""

from collections import defaultdict
from typing import Dict, List, Tuple

from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from torchtyping import TensorType
import torch

from src.processing.alignment import get_char_coverages

# ------------------------------------------------------------------------------------- #

num_frames, num_chars, num_tracks = None, None, None
num_verts, num_faces, num_feats = None, None, None

# ------------------------------------------------------------------------------------- #


def compute_track_scores(
    char_tracks: TensorType["num_tracks", "num_frames", "num_verts", 3],
    faces: TensorType["num_faces", 3],
    w2c_poses: TensorType["num_frames", 4, 4],
    intrinsics: TensorType["num_frames", 3, 3],
) -> TensorType["num_tracks"]:
    """
    Compute track scores based on the provided char tracks, faces, camera poses, and
    intrinsics:
        1. Compute the char coverages for each frame (area of bounding-box).
        2. Compute the track coverages for each track (track length).
        3. Compute the track scores as the product of the two.

    :param char_tracks: tracks of the chars.
    :param faces: faces of the chars.
    :param w2c_poses: camera to world poses.
    :param intrinsics: camera intrinsics.
    :return: track scores.
    """
    # Get camera to world poses
    _rotation = w2c_poses[:, :3, :3]
    _translation = w2c_poses[:, :3, 3]
    c2w_poses = torch.eye(4).repeat(w2c_poses.shape[0], 1, 1)
    c2w_poses[:, :3, :3] = _rotation.mT
    c2w_poses[:, :3, 3] = -(_rotation.mT @ _translation[..., None]).squeeze()

    # Compute track scores
    char_coverages = get_char_coverages(
        char_tracks, faces, w2c_poses, c2w_poses, intrinsics
    )
    normalized_coverages = char_coverages.sum(dim=-1) / (
        (char_coverages > 0).sum(dim=-1) + 1e-9
    )
    num_frames = char_tracks.shape[1]
    track_lengths = (char_tracks.sum(dim=(-2, -1)) != 0.0).sum(-1) / num_frames
    track_scores = normalized_coverages * track_lengths

    return track_scores


def select_main_characters(
    tracks: TensorType["num_tracks", "num_frames", "num_verts", 3],
    scores: TensorType["num_tracks"],
    overlap_threshold: float = 0.0,
    score_threshold: float = 0.01,
) -> List[int]:
    """
    Selects the main characters based on the provided tracks and scores.

    :param tracks: tracks of the characters.
    :param scores: scores of the tracks.
    :param overlap_threshold: threshold for overlap between tracks.
    :param score_threshold: threshold for the track scores.
    :return: indices of the main characters.
    """
    track_scores, track_indices = torch.sort(scores, dim=-1, descending=True)
    adjacency_tracks = tracks.sum(dim=(-2, -1)) != 0.0
    main_characters = []
    remaining_tracks = list(track_indices.numpy())
    while len(remaining_tracks) > 0:
        main_track = remaining_tracks.pop(0)
        if scores[main_track] < score_threshold:
            continue
        main_characters.append(main_track)

        # Check if main_track overlaps with any other tracks and remove them
        to_remove = []
        for k, track_index in enumerate(remaining_tracks):
            overlaps = adjacency_tracks[[main_track, track_index]].sum(dim=0) > 1
            if overlaps.sum() >= overlap_threshold:
                to_remove.append(k)
        remaining_tracks = [x for x in remaining_tracks if x not in to_remove]

    return main_characters


def track_bodies(data: Dict[str, TensorType]) -> Dict[str, TensorType]:
    """
    Track bodies and find main characters.

    :param data: dictionary containing "vertices", "normals", "faces", "pose_se3", and
        "intrinsics".
    :return: the updated input with vertices and normals tracks, and the main character
        track indices.
    """
    # Empty tracks
    if data["vertices"].shape[0] == 0:
        data["vertices_tracks"] = data["vertices"]
        data["main_characters"] = []
        return data

    # Track bodies
    raw_vertices = data["vertices"].permute(1, 0, 2, 3)
    raw_normals = data["normals"].permute(1, 0, 2, 3)
    tracker = SORTTracker(max_age=20, min_hits=2, distance_threshold=2)
    vertices_tracks, normals_tracks = tracker.track(raw_vertices, raw_normals)

    # Find main characters
    track_scores = compute_track_scores(
        vertices_tracks, data["faces"], data["pose_se3"], data["intrinsics"]
    )
    main_characters = select_main_characters(vertices_tracks, track_scores)

    # Update data
    data["vertices_tracks"] = vertices_tracks
    # data["normals_tracks"] = normals_tracks
    data["main_characters"] = main_characters

    return data


class KalmanTracker(object):
    """This class represents the internal state of individual tracked objects."""

    count = 0

    def __init__(
        self,
        char_feats: TensorType["num_feats"],
        char_index: int,
        num_feats: int = 6,
        num_velocity: int = 3,
    ):
        """
        Initializes a tracker using initial char.
        """

        # state x dimension 9 (=6+3): x, y, z, l, w, h, dx, dy, dz
        self.dim_z = num_feats
        self.dim_x = num_feats + num_velocity

        self.kf = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        # State transition (constant velocity: x' = x + dx, y' = y + dy, z' = z + dz)
        self.kf.F = torch.eye(self.dim_x)
        self.kf.F[:num_velocity, -num_velocity:] = torch.eye(num_velocity)
        # Measurement function, dim_z * dim_x
        self.kf.H = torch.zeros((self.dim_z, self.dim_x))
        self.kf.H[:, :num_feats] = torch.eye(num_feats)
        # Uncertainty (given a single data, the initial velocity is very uncertain)
        self.kf.P[num_feats:, num_feats:] *= 1000.0
        self.kf.P *= 10.0
        # Process uncertainty (make the constant velocity part more certain)
        self.kf.Q[num_feats:, num_feats:] *= 0.01
        # Initialize data
        self.kf.x[:num_feats] = char_feats[:, None]

        # State variables
        self.id = KalmanTracker.count
        self.feat_idx = char_index
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []
        KalmanTracker.count += 1

    def update(self, feats: TensorType["num_feats"], index: int):
        """
        Update state variables with observed feats.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(feats)
        self.feat_idx = index

    def predict(self) -> TensorType["num_feats"]:
        """
        Advance state variables and return the predicted feats estimate.
        """
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x)
        return torch.from_numpy(self.history[-1])

    def get_state(self) -> TensorType["num_feats"]:
        """
        Return the current bounding bofeatsx estimate.
        """
        return self.kf.x


class SORTTracker(object):
    def __init__(
        self, max_age: int = 15, min_hits: int = 2, distance_threshold: float = 2
    ):
        """
        Sets key parameters for SORT.add()

        :param max_age: max nb of frames a tracklet can be inactive before being removed.
        :param min_hits: min nb of hits required for a tracklet to be considered valid.
        :param distance_threshold: threshold distance (meters) for matching detections to
            existing tracklets.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0

    # Tracking helper methods
    # --------------------------------------------------------------------------------- #

    def update(
        self, char_feats: TensorType["num_chars", "num_feats"]
    ) -> List[Tuple[int, int]]:
        """Perform 1 tracking step."""
        num_feats = char_feats.shape[1]
        self.frame_count += 1

        # Get predicted locations from existing trackers.
        tracks = torch.zeros((len(self.trackers), num_feats))
        for t, track in enumerate(tracks):
            pos = self.trackers[t].predict()
            track[:] = pos[:num_feats].squeeze()

        # Match existing trackers to detections
        matched, unmatched_feats, unmatched_tracks = self.match_featandtracks(
            char_feats, tracks, self.distance_threshold
        )

        # Update matched trackers with assigned feats
        for m in matched:
            self.trackers[m[1]].update(char_feats[m[0], :], m[0])

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_feats:
            track = KalmanTracker(char_feats[i, :], i)
            self.trackers.append(track)

        ret, i = [], len(self.trackers)
        for tracker_index, track in reversed([x for x in enumerate(self.trackers)]):
            if (track.time_since_update < 1) and (
                track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append((track.feat_idx, track.id))
            i -= 1
            # Remove dead tracklet
            if track.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return ret

        return []

    @staticmethod
    def match_featandtracks(
        feats: TensorType["num_chars", "num_feats"],
        trackers: TensorType["num_chars", "num_feats"],
        distance_threshold: float,
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """
        Assign detections to tracked object (both represented as bounding boxes).
        """
        num_feats = feats.shape[1]

        # Empty tracker/buffer
        if len(trackers) == 0:
            return (
                torch.empty((0, 2), dtype=int),
                torch.arange(len(feats)),
                torch.empty((0, num_feats), dtype=int),
            )
        # Compute disatnce matrix
        cdist_matrix = torch.cdist(feats[..., :3], trackers[..., :3])

        # Assign matches
        if min(cdist_matrix.shape) > 0:
            a = (cdist_matrix < distance_threshold).to(bool)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = torch.stack(torch.where(a), axis=1)
            else:
                if torch.isinf(cdist_matrix.sum()) or torch.isnan(cdist_matrix.sum()):
                    matched_indices = torch.empty((0, 2))
                else:
                    try:
                        x, y = linear_sum_assignment(cdist_matrix)
                    except:
                        import ipdb

                        ipdb.sset_trace()
                    matched_indices = torch.tensor(list(zip(x, y)))
        else:
            matched_indices = torch.empty((0, 2))

        # Find unmatched detections and trackers
        unmatched_feats = []
        for d, _ in enumerate(feats):
            if d not in matched_indices[:, 0]:
                unmatched_feats.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matched with high distances
        matches = []
        for m in matched_indices:
            if cdist_matrix[m[0], m[1]] > distance_threshold:
                unmatched_feats.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = torch.empty((0, 2), dtype=int)
        else:
            matches = torch.cat(matches, dim=0)

        return matches, torch.tensor(unmatched_feats), torch.tensor(unmatched_trackers)

    @staticmethod
    def process_chars(
        raw_chars: TensorType["num_chars", "num_verts", 3],
        raw_normals: TensorType["num_chars", "num_verts", 3],
    ) -> TensorType["num_chars", "num_feats"]:
        """
        Process the raw chars and compute char features.
        i.e.: center (3d), width (1d), height (1d), length (1d).

        NOTE: Could add char orientation as well.
        """
        char_centers = raw_chars.mean(dim=1)
        char_width = raw_chars.max(1).values[:, 0] - raw_chars.min(1).values[:, 0]
        char_height = raw_chars.max(1).values[:, 1] - raw_chars.min(1).values[:, 1]
        char_length = raw_chars.max(1).values[:, 2] - raw_chars.min(1).values[:, 2]

        char_feats = torch.cat(
            [
                char_centers,
                char_width[:, None],
                char_height[:, None],
                char_length[:, None],
            ],
            dim=1,
        )

        return char_feats

    # Interpolation method
    # --------------------------------------------------------------------------------- #

    @staticmethod
    def find_missedframes(
        verts: TensorType["num_frames", "num_verts", 3]
    ) -> List[List[int]]:
        """Find the indices of missed frames in a sequence of vertices."""
        detections = verts.sum([1, 2]) > 0
        buffer, missed_indices = [], []
        start_true = False
        for i, detection in enumerate(detections):
            start_true = True if detection else start_true
            if (not detection) and start_true:
                buffer.append(i)
            elif start_true and detection and buffer:
                missed_indices.append(buffer)
                buffer = []
        return missed_indices

    def interpolate(
        self, verts: TensorType["num_frames", "num_verts", 3]
    ) -> TensorType["num_frames", "num_verts", 3]:
        """
        Interpolate missing frames in a sequence of vertices.
        """
        missed_indices = self.find_missedframes(verts)
        for missed in missed_indices:
            start_index, end_index = missed[0] - 1, missed[-1] + 1
            start_verts, end_verts = verts[start_index], verts[end_index]
            interpolation_rates = torch.arange(len(missed) + 2) / (len(missed) + 1)
            diff_verts = end_verts - start_verts
            for k, index in enumerate(range(start_index, end_index)):
                verts[index] = start_verts + diff_verts * interpolation_rates[k]
        return verts

    # Main tracking method
    # --------------------------------------------------------------------------------- #

    def track(
        self,
        raw_chars: TensorType["num_frames", "num_chars", "num_verts", 3],
        raw_normals: TensorType["num_frames", "num_chars", "num_verts", 3],
    ) -> Tuple[
        TensorType["num_tracks", "num_frames", "num_verts", 3],
        TensorType["num_tracks", "num_frames", "num_verts", 3],
    ]:
        """
        Main tracking method. It processes raw chars and normals, updates tracks, and
        interpolates the missing frames.
        """
        # Track chars
        num_frames, _, num_verts, _ = raw_normals.shape
        _tracks = defaultdict(list)
        for frame_index in range(num_frames):
            char_feats = self.process_chars(
                raw_chars[frame_index], raw_normals[frame_index]
            )
            null_mask = char_feats.sum(dim=1) == 0
            track_indices = self.update(char_feats[~null_mask])
            for char_index, track_index in track_indices:
                _tracks[track_index].append(
                    (
                        frame_index,
                        raw_chars[frame_index, ~null_mask][char_index],
                        raw_normals[frame_index, ~null_mask][char_index],
                    )
                )
        # Reset track indices to avoid sparse indices
        _tracks = {k: v for k, v in enumerate(_tracks.values())}

        # Post-process tracks
        char_tracks = torch.zeros((len(_tracks), num_frames, num_verts, 3))
        normals_tracks = torch.zeros((len(_tracks), num_frames, num_verts, 3))
        for track_index, track in _tracks.items():
            for frame_index, char, normals in track:
                try:
                    char_tracks[track_index, frame_index] = char
                    normals_tracks[track_index, frame_index] = normals
                except:
                    import ipdb

                    ipdb.sset_trace()

            char_tracks[track_index] = self.interpolate(char_tracks[track_index])
            normals_tracks[track_index] = self.interpolate(normals_tracks[track_index])

        return char_tracks, normals_tracks
