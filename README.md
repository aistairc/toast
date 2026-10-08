# TOAST

<p align="center">
  <a href="https://arxiv.org/abs/2610.00899">Paper</a>&nbsp | <a href="https://kskshr.github.io/toast/">Project Page</a>&nbsp | <a href="https://huggingface.co/aist-eart/toast">Tokenizer</a>
</p>

This is a official implementation of TOAST. This repository is built on [openpi](https://github.com/Physical-Intelligence/openpi): TOAST replaces the FAST tokenizer of the π₀-FAST model. For installation, requirements and the general usage of openpi (data conversion, training, serving, PyTorch support, troubleshooting), see [README_openpi.md](README_openpi.md).


## Setup

Follow the [installation instructions of openpi](README_openpi.md#installation):

```bash
git clone --recurse-submodules https://github.com/aistairc/toast.git
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

Building a vocabulary from DROID additionally needs the RLDS data loader:

```bash
uv sync --group rlds
```

## 1. Building a vocabulary

A TOAST tokenizer is fitted on a corpus of normalized action chunks, in three steps: compute the norm stats of the
corpus, extract action chunks, and fit the subword model.

The example below builds the tokenizer used in our LIBERO experiments: 512 subwords over DCT symbols (scale 10,
dimension-wise), fitted on 1M end-effector action chunks from DROID.

```bash
# DROID in RLDS format: the parent directory of the `droid` directory (see examples/droid/README_train.md).
export RLDS_DATA_DIR=/path/to/rlds

# 1. Norm stats of the corpus.
uv run --group rlds scripts/compute_norm_stats.py --config-name droid_eef_action_chunks --max-frames 5000000

# 2. Extract 1M normalized action chunks.
uv run --group rlds scripts/extract_action_chunks.py --config-name droid_eef_action_chunks \
    --output-path tokenizer/action_chunks/droid_eef.h5 --num-chunks 1000000

# 3. Fit the subword model on the extracted chunks (see "Building Tokenizer" at https://huggingface.co/aist-eart/toast).
uv run python - <<'EOF'
import h5py
from transformers import AutoProcessor

with h5py.File("tokenizer/action_chunks/droid_eef.h5") as f:
    action_chunks = f["action_chunks"][:]
tokenizer = AutoProcessor.from_pretrained("aist-eart/toast", trust_remote_code=True)
tokenizer.fit(action_chunks, scale=10, vocab_size=512).save_pretrained(
    "tokenizer/models/droid_eef_unigram_dct-scale-10_vocab-512_rowwise"
)
EOF
```

The result is the released tokenizer [`aist-eart/toast`](https://huggingface.co/aist-eart/toast), which can be used
directly as `toast_action_tokenizer_dir` instead of rebuilding it.

**Using your own dataset.** `scripts/extract_action_chunks.py` works with any openpi training config, so a
vocabulary can be built from any dataset you can train on (see
[README_openpi.md](README_openpi.md#fine-tuning-base-models-on-your-own-data) for defining a config). Chunks of
several corpora can be concatenated before `fit`. The options of `fit` (quantization, order, subword model) are
documented in the [tokenizer's README](https://huggingface.co/aist-eart/toast); the settings of the tokenizers used
in our experiments are listed in
[`src/openpi/training/misc/toast_config.py`](src/openpi/training/misc/toast_config.py).

## 2. Training

TOAST is enabled through the model config: when `toast_action_tokenizer_dir` is set, the tokenizer in that directory
replaces FAST.

```python
TrainConfig(
    name="my_toast_config",
    model=pi0_fast.Pi0FASTConfig(
        action_dim=7,
        action_horizon=10,
        max_token_len=180,
        toast_action_tokenizer_dir="tokenizer/models/droid_eef_unigram_dct-scale-10_vocab-512_rowwise",
        sample_segmentation=True,  # True: TOAST, False: TOAST (deterministic)
        alpha=0.1,
        nbest_size=64,
    ),
    data=...,  # as for any other openpi config
    weight_loader=weight_loaders.PaliGemmaWeightLoader(),
)
```

Training then works as for any other openpi config:

```bash
# Norm stats of the training dataset (requires the tokenizer to exist).
uv run scripts/compute_norm_stats.py --config-name my_toast_config

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py my_toast_config --exp-name=my_experiment --overwrite
```

Ready-made configs for LIBERO are defined in
[`src/openpi/training/misc/toast_config.py`](src/openpi/training/misc/toast_config.py):

| Config | Method | Dataset |
|---|---|---|
| `pi0_toast_libero` | TOAST | `physical-intelligence/libero` |
| `pi0_toast_deterministic_libero` | TOAST (deterministic) | `physical-intelligence/libero` |
| `pi0_toast_libero_1of{2,4,8,16}` | TOAST | 1/2, 1/4, 1/8, 1/16 of the LIBERO demonstrations |
| `pi0_toast_deterministic_libero_1of{2,4,8,16}` | TOAST (deterministic) | 1/2, 1/4, 1/8, 1/16 of the LIBERO demonstrations |

The same configs exist for the binning and BEAST quantizations, e.g. `pi0_toast_binning_libero` and
`pi0_toast_beast_deterministic_libero_1of16`.

All of them fine-tune from the PaliGemma weights and predict actions relative to the start of the chunk.

## 3. Evaluation

A TOAST checkpoint is served like any other openpi checkpoint. The tokenizer directory is read from the path in the
config, relative to the working directory.

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_toast_libero \
    --policy.dir=checkpoints/pi0_toast_libero/my_experiment/29999
```

Then run the evaluation client of your environment against the server; see
[README_openpi.md](README_openpi.md#3-spinning-up-a-policy-server-and-running-inference) and
[docs/remote_inference.md](docs/remote_inference.md). For LIBERO, see the next section.

## Reproducing the LIBERO results

**1. Vocabulary.** Build the DROID vocabulary as described in [1. Building a vocabulary](#1-building-a-vocabulary).

**2. Data.** The configs use the LeRobot dataset `physical-intelligence/libero` (converted from the RLDS release with
[`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py), see
[README_openpi.md](README_openpi.md#1-convert-your-data-to-a-lerobot-dataset)) and, for the low-resource settings, its subsets
`<user>/libero_1of{2,4,8,16}`: a fixed fraction of the demonstrations of every task suite, selected with a fixed seed.
They are built from the converted full dataset with

```bash
for scale in 2 4 8 16; do
    uv run examples/libero/subset_libero_lerobot.py --scale ${scale}   # -> <user>/libero_1of${scale}
done
```

**3. Norm stats.** One per dataset, shared by both methods:

```bash
for dataset in libero libero_1of2 libero_1of4 libero_1of8 libero_1of16; do
    uv run scripts/compute_norm_stats.py --config-name pi0_toast_${dataset}
done
```

**4. Training.** 30k steps with batch size 32:

```bash
for config in pi0_toast_libero pi0_toast_deterministic_libero; do  # or ..._libero_1of{2,4,8,16}
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py ${config} \
        --exp-name=seed-0 --seed=0 --batch-size=32 --num-train-steps=30000 \
        --save-interval=30000 --keep-period=30000 --overwrite
done
```

**5. Evaluation.** Set up the LIBERO simulation environment as described in
[examples/libero/README.md](examples/libero/README.md), then:

```bash
# Terminal 1: policy server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_toast_libero \
    --policy.dir=checkpoints/pi0_toast_libero/seed-0/29999

# Terminal 2: LIBERO client, once per task suite
for suite in libero_spatial libero_object libero_goal libero_10; do
    python examples/libero/main.py --args.task-suite-name ${suite}
done
```

**Reproduced results.** For reference, average success rates (%) over the four LIBERO suites obtained with this
repository: vocabulary rebuilt from DROID as above, one training run (seed 0) per setting, 50 rollouts per task
(2,000 episodes per entry).

| Method | 1/1 | 1/2 | 1/4 | 1/8 | 1/16 |
|---|---|---|---|---|---|
| TOAST (deterministic) | 91.1 | 84.4 | 72.8 | 55.6 | 38.9 |
| TOAST | 93.2 | 86.0 | 78.5 | 59.9 | 48.1 |


## Citation

```bibtex
@misc{shirai2026toaststochasticrobotaction,
      title={TOAST: Stochastic Robot Action Tokenization for Autoregressive Vision-Language-Action Models}, 
      author={Keisuke Shirai and Tomohiro Motoda and Hanbit Oh and Ryoichi Nakajo and Roman Mykhailyshyn and Ryo Hanai and Shotaro Miwa and Yukiyasu Domae},
      year={2026},
      eprint={2610.00899},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.00899}, 
}
```


## Acknowledgements

This repository is based on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical Intelligence. See
[README_openpi.md](README_openpi.md) for the original README and [LICENSE](LICENSE) for the license.


