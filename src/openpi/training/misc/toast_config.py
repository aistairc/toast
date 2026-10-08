import os

import openpi.models.pi0_fast as pi0_fast
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

# TOAST tokenizers fitted on 1M DROID end-effector action chunks (tokenizer/action_chunks/droid_eef.h5, see the
# README), one per quantization. All of them are unigram vocabularies over symbols flattened dimension-wise
# (dimension-major), built with `fit()` of https://huggingface.co/aist-eart/toast as follows.
DROID_EEF_TOKENIZER_DIRS = {
    # DCT coefficients (scale 10), 512 subwords. The default TOAST tokenizer, released as aist-eart/toast:
    #   fit(action_chunks, scale=10, vocab_size=512)
    "dct": "tokenizer/models/droid_eef_unigram_dct-scale-10_vocab-512_rowwise",
    # 256 uniform bins decoded to their centers, 1024 subwords:
    #   fit(action_chunks, vocab_size=1024, quantization="binning", quantizer_kwargs={"bin_size": 256, "use_bin_center": True})
    "binning": "tokenizer/models/droid_eef_unigram_binning-center-binsize-256_vocab-1024_dimension",
    # BEAST: 5 B-spline control points per dimension (zero-order spline for the gripper) with 256 levels, 512 subwords:
    #   fit(action_chunks, vocab_size=512, quantization="beast", quantizer_kwargs={"num_basis": 5, "gripper_zero_order": True})
    "beast": "tokenizer/models/droid_eef_unigram_beast-basis-5_vocab-512_dimension",
}


def get_toast_configs():
    # Import here to avoid circular imports.
    from openpi.training.config import AssetsConfig
    from openpi.training.config import DataConfig
    from openpi.training.config import LeRobotLiberoDataConfig
    from openpi.training.config import RLDSDroidDataConfig
    from openpi.training.config import TrainConfig

    def droid_eef_action_chunks(name: str, *, relative_transform: bool) -> TrainConfig:
        # Not meant for training: these configs define the DROID end-effector action chunks (7 dims: 6D pose +
        # gripper, 16 steps) the tokenizer vocabulary is fitted on. Use them with scripts/compute_norm_stats.py
        # and scripts/extract_action_chunks.py.
        return TrainConfig(
            name=name,
            model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=16, max_token_len=180),
            data=RLDSDroidDataConfig(
                repo_id="droid",
                # Path to your DROID RLDS dataset (the parent directory of the `droid` directory).
                rlds_data_dir=os.environ.get("RLDS_DATA_DIR", "<path_to_droid_rlds_dataset>"),
                action_space=droid_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION,
                relative_transform=relative_transform,
            ),
            batch_size=256,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
        )

    def toast_libero(quantization: str, dataset: str, repo_id: str, *, sample_segmentation: bool) -> TrainConfig:
        # pi0_toast[_binning|_beast][_deterministic]_<dataset>
        name = "pi0_toast" + ("" if quantization == "dct" else f"_{quantization}")
        name += "" if sample_segmentation else "_deterministic"
        return TrainConfig(
            name=f"{name}_{dataset}",
            model=pi0_fast.Pi0FASTConfig(
                action_dim=7,
                action_horizon=10,
                max_token_len=180,
                toast_action_tokenizer_dir=DROID_EEF_TOKENIZER_DIRS[quantization],
                sample_segmentation=sample_segmentation,
                alpha=0.1,
                nbest_size=64,
            ),
            data=LeRobotLiberoDataConfig(
                repo_id=repo_id,
                base_config=DataConfig(prompt_from_task=True),
                # LIBERO actions are step-to-step deltas; train on actions relative to the start of the chunk.
                extra_delta_transform=False,
                delta_to_relative_transform=True,
                # All methods share the norm stats computed with the `pi0_toast_<dataset>` config.
                assets=AssetsConfig(assets_dir=f"./assets/pi0_toast_{dataset}"),
            ),
            # Start from the PaliGemma weights (no robot pre-training).
            weight_loader=weight_loaders.PaliGemmaWeightLoader(),
            lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, decay_steps=29_000),
            num_train_steps=30_000,
        )

    return [
        #
        # Tokenizer corpus configs.
        #
        # Absolute end-effector pose actions.
        droid_eef_action_chunks("droid_eef_action_chunks", relative_transform=False),
        # End-effector pose actions relative to the pose at the start of the chunk.
        droid_eef_action_chunks("droid_eef_relative_action_chunks", relative_transform=True),
        #
        # Fine-tuning configs for LIBERO, and for its low-resource subsets (1/2 ... 1/16 of the demonstrations):
        # pi0_toast[_binning|_beast][_deterministic]_libero[_1of{2,4,8,16}].
        #
        *[
            toast_libero(quantization, dataset, repo_id, sample_segmentation=sample_segmentation)
            for quantization in DROID_EEF_TOKENIZER_DIRS
            for dataset, repo_id in [
                ("libero", "physical-intelligence/libero"),
                *[(f"libero_1of{n}", f"kskshr/libero_1of{n}") for n in (2, 4, 8, 16)],
            ]
            # TOAST samples the subword segmentation of the action tokens during training; TOAST (deterministic)
            # always uses the most likely segmentation.
            for sample_segmentation in (True, False)
        ],
    ]


