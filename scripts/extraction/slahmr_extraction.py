from pathlib import Path
from omegaconf import DictConfig

from evo.core.trajectory import PosePath3D
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import hydra

from helper.files import save_frames
from lib.slahmr.slahmr.data.dataset import load_cameras_npz
from src.data.cm_dataset import CMDataset
from src.logger import Logger
from src.slahmr.phalp import run_phalp
from src.slahmr.slahmr import run_slahmr
from src.slahmr.slam import run_slam

# ------------------------------------------------------------------------------------- #

MIN_FRAMES = 5 * 25  # Skip shot shorter than 5 seconds
SLAM_THRESHOLD = 1.0  # m

# ------------------------------------------------------------------------------------- #


def check_slam(slam_file, threshold):
    cam_R, cam_t, intrinsics, _, _ = load_cameras_npz(slam_file)

    num_cams = cam_R.shape[0]
    pose_se3 = torch.eye(4).unsqueeze(0).repeat((num_cams, 1, 1))
    pose_se3[:, :3, :3] = cam_R
    pose_se3[:, :3, 3] = cam_t

    slam_length = PosePath3D(poses_se3=pose_se3.numpy()).get_infos()["path length (m)"]
    if slam_length < threshold:
        return False, slam_length
    else:
        return True, slam_length


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(config: DictConfig) -> None:
    dataset = CMDataset(**config.data)
    dataloader = DataLoader(dataset)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    clip_path = Path(config.data.data_dir) / "metadata" / config.data.metadata_filename
    ref_path = Path(config.data.data_dir) / "metadata" / config.data.ref_filename
    logger = Logger(save_dir)

    for batch in tqdm(dataloader):
        frames, metadata = batch

        year = str(int(float(metadata["upload_year"].item())))
        videoid = metadata["videoid"][0]
        shot_index = str(metadata["shot_index"].item())
        file_dir = save_dir / year / videoid / shot_index.zfill(5)
        shot_id = "-".join([year, videoid, shot_index])
        file_dir.mkdir(exist_ok=True, parents=True)

        if frames.shape[1] == 0 or frames is None:
            logger.missing_data(shot_id, -1)
            logger.update_ref(clip_path, ref_path)
            continue

        # Save all frames
        (file_dir / "images").mkdir(exist_ok=True, parents=True)
        frames = frames.squeeze(0).permute(0, 3, 1, 2)
        save_frames(frames, file_dir / "images")
        num_frames = frames.shape[0]
        if num_frames < MIN_FRAMES:
            logger.skipping_data(shot_id, num_frames)
            logger.update_ref(clip_path, ref_path)
            continue

        # Run droid slam
        logger.run(run_slam, shot_id, file_dir, num_frames, **config.model.slam)
        slam_file = file_dir / "slam_out" / "cameras.npz"
        if slam_file.exists():
            slam_check, slam_length = check_slam(slam_file, SLAM_THRESHOLD)
            if not slam_check:
                logger.skipping_data(
                    shot_id, num_frames, f"Slam too short: {slam_length:.2f} m"
                )
                logger.update_ref(clip_path, ref_path)
                continue

        # Run PHALP
        logger.run(run_phalp, shot_id, file_dir, num_frames, **config.model.phalp)

        # Run slahmr
        logger.run(run_slahmr, shot_id, file_dir, num_frames, **config.model.slahmr)

        logger.set_buffer()


if __name__ == "__main__":
    main()
