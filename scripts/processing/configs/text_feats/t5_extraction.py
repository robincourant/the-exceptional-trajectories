import argparse
from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer, T5EncoderModel
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
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch_size", "-bs", type=int, default=50)
    parser.add_argument("--max_token_length", "-tl", type=int, default=512)
    parser.add_argument("--device", "-s", type=str, default="cuda")
    parser.add_argument("--t5_version", "-tv", type=str, default="google/flan-t5-xl")
    args = parser.parse_args()

    return args.__dict__


# ------------------------------------------------------------------------------------- #


def get_samplepaths(data_dir: Path) -> List[str]:
    sample_paths = [str(sample_path) for sample_path in sorted(data_dir.iterdir())]
    return sample_paths


def save_feats(
    feats: TensorType["batch_size", "context_length"],
    masks: TensorType["batch_size"],
    sample_paths: List[str],
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for feat, mask, sample_path in zip(feats, masks, sample_paths):
        output_path = output_dir / (Path(sample_path).stem + ".pth")
        torch.save(feat[:mask], output_path)


# ------------------------------------------------------------------------------------- #


def load_t5_model(version: str, device: str) -> Tuple[AutoTokenizer, T5EncoderModel]:
    tokenizer = AutoTokenizer.from_pretrained(version)

    model = T5EncoderModel.from_pretrained(version)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return tokenizer, model


@torch.no_grad()
def encode_text(
    caption_raws: List[str],  # batch_size
    tokenizer: AutoTokenizer,
    model: T5EncoderModel,
    max_token_length: int,
    device: str,
) -> Tuple[TensorType["batch_size", "context_length"], TensorType["batch_size"]]:
    texts = tokenizer(
        caption_raws,
        return_tensors="pt",
        padding="longest",
        max_length=max_token_length,
        truncation=True,
    ).to(device)
    x = model(**texts).last_hidden_state.detach()

    return x.cpu(), texts["attention_mask"].sum(dim=1).cpu()


# ------------------------------------------------------------------------------------- #


def extract_t5(
    input_dir: Path,
    output_dir: Path,
    batch_size: int,
    max_token_length: int,
    t5_version: str,
    device: str,
):
    caption_paths = get_samplepaths(input_dir)
    caption_loader = DataLoader(caption_paths, batch_size=batch_size, shuffle=False)
    tokenizer, t5_model = load_t5_model(t5_version, device)

    with PROGRESS:
        task = PROGRESS.add_task(
            "[green]Processing...", total=len(caption_loader), step=0
        )
        for caption_path_batch in caption_loader:
            caption_raws = [load_txt(x) for x in caption_path_batch]
            caption_feats, caption_masks = encode_text(
                caption_raws, tokenizer, t5_model, max_token_length, device
            )
            save_feats(caption_feats, caption_masks, caption_path_batch, output_dir)
            PROGRESS.update(task, advance=1)


if __name__ == "__main__":
    args = parse_args()
    extract_t5(**args)
