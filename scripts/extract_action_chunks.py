"""Extract normalized action chunks from an existing dataset, as the corpus for building a TOAST tokenizer.

The chunks go through the same repack / data / normalization transforms as during training with the given config,
i.e. they are exactly what the action tokenizer sees. The norm stats of the config must exist already (see
scripts/compute_norm_stats.py). The result is an HDF5 file with a single dataset `action_chunks` of shape
[num_chunks, action_horizon, action_dim], which is the corpus the tokenizer is fitted on (`fit()` of https://huggingface.co/aist-eart/toast).

Examples:
    # 1M chunks from DROID (RLDS). RLDS_DATA_DIR is the parent directory of the `droid` directory.
    RLDS_DATA_DIR=/path/to/rlds uv run --group rlds scripts/extract_action_chunks.py \
        --config-name droid_eef_action_chunks --output-path tokenizer/action_chunks/droid_eef.h5 --num-chunks 1000000

    # All chunks from LIBERO (LeRobot).
    uv run scripts/extract_action_chunks.py \
        --config-name pi0_toast_libero --output-path tokenizer/action_chunks/libero_relative.h5
"""

import logging
import pathlib

import h5py
import numpy as np
import torch
import tqdm
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms


class _ActionsOnly(_transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {"actions": np.asarray(x["actions"], dtype=np.float32)}


def _input_transforms(data_config: _config.DataConfig) -> list[_transforms.DataTransformFn]:
    if data_config.norm_stats is None:
        raise ValueError(
            "Normalization stats not found. Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
        )
    # Same as for training, but without the model transforms (tokenization, image resizing, padding).
    return [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        _ActionsOnly(),
    ]


def _iter_lerobot(config: _config.TrainConfig, data_config: _config.DataConfig, num_chunks: int, args):
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.TransformedDataset(dataset, _input_transforms(data_config))

    if num_chunks <= 0 or num_chunks >= len(dataset):
        num_chunks = len(dataset)
    else:
        indices = np.random.default_rng(args["seed"]).choice(len(dataset), size=num_chunks, replace=False)
        dataset = torch.utils.data.Subset(dataset, np.sort(indices).tolist())

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args["batch_size"], num_workers=args["num_workers"], shuffle=False
    )
    return (np.asarray(batch["actions"]) for batch in loader), num_chunks


def _iter_rlds(config: _config.TrainConfig, data_config: _config.DataConfig, num_chunks: int, args):
    if num_chunks <= 0:
        raise ValueError("--num-chunks must be set for RLDS datasets (they are sampled from an infinite stream).")
    from openpi.training.droid_rlds_dataset import DroidRldsDataset

    dataset = DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=args["batch_size"],
        datasets=data_config.datasets,
        shuffle=True,
        action_chunk_size=config.model.action_horizon,
        action_space=data_config.action_space,
        shuffle_buffer_size=args["shuffle_buffer_size"],
    )
    dataset = _data_loader.IterableTransformedDataset(dataset, _input_transforms(data_config), is_batched=True)
    return (batch["actions"] for batch in dataset), num_chunks


def main(
    config_name: str,
    output_path: str,
    num_chunks: int = -1,
    batch_size: int = 256,
    num_workers: int = 8,
    shuffle_buffer_size: int = 250_000,
    seed: int = 0,
):
    """Extract action chunks.

    Args:
        config_name: Training config that defines the dataset and its transforms.
        output_path: HDF5 file to write.
        num_chunks: Number of chunks to extract. -1 extracts every chunk (LeRobot datasets only); a smaller number
            extracts a random subset.
        batch_size: Batch size used for loading.
        num_workers: Number of data loader workers (LeRobot datasets only).
        shuffle_buffer_size: Size of the shuffle buffer (RLDS datasets only). Reduce it if you run out of memory.
        seed: Seed for choosing the random subset (LeRobot datasets only).
    """
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle_buffer_size": shuffle_buffer_size,
        "seed": seed,
    }

    if data_config.rlds_data_dir is not None:
        batches, num_chunks = _iter_rlds(config, data_config, num_chunks, args)
    else:
        batches, num_chunks = _iter_lerobot(config, data_config, num_chunks, args)

    action_chunks = np.zeros((num_chunks, config.model.action_horizon, config.model.action_dim), dtype=np.float32)
    count = 0
    with tqdm.tqdm(total=num_chunks, desc="Extracting action chunks") as pbar:
        for batch in batches:
            actions = batch[: num_chunks - count]
            action_chunks[count : count + len(actions)] = actions
            count += len(actions)
            pbar.update(len(actions))
            if count >= num_chunks:
                break
    if count < num_chunks:
        raise RuntimeError(f"Dataset ran out after {count} chunks, expected {num_chunks}.")

    path = pathlib.Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("action_chunks", data=action_chunks)
    logging.info(
        f"Wrote {action_chunks.shape} action chunks to {path} (min {action_chunks.min():.3f}, max {action_chunks.max():.3f})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
