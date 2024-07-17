import argparse
from pathlib import Path
from typing import List

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchtyping import TensorType

from helper.files import load_txt
from helper.progress import PROGRESS

# ------------------------------------------------------------------------------------- #

batch_size, context_length = None, None

# ------------------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--batch_size", "-bs", type=int, default=50)
    parser.add_argument("--max_token_length", "-tl", type=int, default=None)
    parser.add_argument("--save_seq", "-sq", action="store_true")
    parser.add_argument("--save_token", "-t", action="store_true")
    parser.add_argument("--save_mask", "-m", action="store_true")
    parser.add_argument("--device", "-s", type=str, default="cuda")
    parser.add_argument("--clip_version", "-cv", type=str, default="ViT-B/32")
    args = parser.parse_args()

    return args.__dict__


# ------------------------------------------------------------------------------------- #


def get_samplepaths(data_dir: Path) -> List[str]:
    sample_paths = [str(sample_path) for sample_path in sorted(data_dir.iterdir())]
    return sample_paths


def save_feats(
    feats: TensorType["batch_size", "context_length"],
    sample_paths: List[str],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for feat, sample_path in zip(feats, sample_paths):
        output_path = output_dir / (Path(sample_path).stem + ".npy")
        np.save(output_path, feat.cpu().numpy())


# ------------------------------------------------------------------------------------- #


def load_clip_model(version: str, device: str) -> clip.model.CLIP:
    model, _ = clip.load(version, device=device, jit=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def encode_text(
    caption_raws: List[str],  # batch_size
    clip_model: clip.model.CLIP,
    max_token_length: int,
    device: str,
) -> TensorType["batch_size", "context_length"]:
    if max_token_length is not None:
        default_context_length = 77
        context_length = max_token_length + 2  # start_token + 20 + end_token
        assert context_length < default_context_length
        # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = clip.tokenize(
            caption_raws, context_length=context_length, truncate=True
        )
        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype,
            device=texts.device,
        )
        texts = torch.cat([texts, zero_pad], dim=1)
    else:
        # [bs, context_length] # if n_tokens > 77 -> will truncate
        texts = clip.tokenize(caption_raws, truncate=True)

    # [batch_size, n_ctx, d_model]
    x = clip_model.token_embedding(texts.to(device)).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)
    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest in each sequence)
    x_tokens = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)].float()
    x_seq = [x[k, : (m + 1)].float() for k, m in enumerate(texts.argmax(dim=-1))]

    return x_seq, x_tokens


# ------------------------------------------------------------------------------------- #


def extract_clip(
    data_dir: Path,
    batch_size: int,
    max_token_length: int,
    clip_version: str,
    save_seq: bool,
    save_token: bool,
    save_mask: bool,
    device: str,
):
    caption_paths = get_samplepaths(data_dir / "caption")
    caption_loader = DataLoader(caption_paths, batch_size=batch_size, shuffle=False)
    output_dir = data_dir / "caption_clip"
    clip_model = load_clip_model(clip_version, device)

    with PROGRESS:
        task = PROGRESS.add_task(
            "[green]Processing...", total=len(caption_loader), step=0
        )
        for caption_path_batch in caption_loader:
            caption_raws = [load_txt(x) for x in caption_path_batch]
            caption_feats = encode_text(
                caption_raws, clip_model, max_token_length, device
            )
            caption_seq, caption_token = caption_feats

            if save_seq:
                save_feats(caption_seq, caption_path_batch, output_dir / "seq")
            if save_token:
                save_feats(caption_token, caption_path_batch, output_dir / "token")
            PROGRESS.update(task, advance=1)


if __name__ == "__main__":
    args = parse_args()
    extract_clip(**args)
