"""Build the 1/n LIBERO subsets (``kskshr/libero_1of{2,4,8,16}``) from the converted full LIBERO dataset.

The low-resource configs (``pi0_toast*_libero_1of{n}``) train on a fixed fraction of the demonstrations of
``physical-intelligence/libero``. The selection is per task suite: the four suites are taken in the order of the
converted dataset (``libero_10, libero_goal, libero_object, libero_spatial``), the episodes of a suite are shuffled with
``random.Random(seed)`` and the first ``int(num_episodes / scale)`` are kept. With the default seed this reproduces
the subsets of our experiments (1/2: 846 episodes, 1/4: 422, 1/8: 210, 1/16: 104).

The subset is written next to the source dataset in the LeRobot cache (``~/.cache/huggingface/lerobot``); the
parquet files are copied with renumbered episode / frame indices, so no re-conversion from RLDS is needed.

Usage:
  uv run examples/libero/subset_libero_lerobot.py --scale 8                   # -> kskshr/libero_1of8
  uv run examples/libero/subset_libero_lerobot.py --scale 8 --dst-repo-id my_user/libero_1of8
"""

import json
import pathlib
import random
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

# Tasks per suite in the LIBERO benchmark; the converter (convert_libero_data_to_lerobot.py) appends the suites one
# after the other, so every ten consecutive task indices are one suite.
TASKS_PER_SUITE = 10


def _episode_parquet_path(root: pathlib.Path, data_path_tmpl: str, ep_idx: int, chunks_size: int) -> pathlib.Path:
    return root / data_path_tmpl.format(episode_chunk=ep_idx // chunks_size, episode_index=ep_idx)


def _select_episodes(episodes: list[dict], scale: int, seed: int) -> list[int]:
    """Indices of the episodes to keep: a shuffled fraction of every suite."""
    # Suite of an episode = index of its task in first-seen order, divided by ten.
    task_order: dict[str, int] = {}
    suite_of_episode = []
    for ep in episodes:
        task = ep["tasks"][0]
        if task not in task_order:
            task_order[task] = len(task_order)
        suite_of_episode.append(task_order[task] // TASKS_PER_SUITE)

    selected: list[int] = []
    for suite in sorted(set(suite_of_episode)):
        block = [i for i, s in enumerate(suite_of_episode) if s == suite]
        # Shuffle positions *within* the suite and map back to global episode indices.
        local = list(range(len(block)))
        random.Random(seed).shuffle(local)
        keep = int(len(block) * (1.0 / scale))
        if keep == 0:
            raise ValueError(f"scale={scale} leaves 0 episodes in suite {suite} of {len(block)}")
        print(f"  suite {suite}: {keep}/{len(block)} episodes")
        selected.extend(block[i] for i in local[:keep])
    return selected


def main(
    src_repo_id: str = "physical-intelligence/libero",
    *,
    scale: int = 8,
    seed: int = 0,
    dst_repo_id: str | None = None,
    overwrite: bool = False,
):
    if scale < 2:
        raise ValueError(f"scale must be >= 2, got {scale}")

    src_root = HF_LEROBOT_HOME / src_repo_id
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    dst_repo_id = dst_repo_id or f"kskshr/libero_1of{scale}"
    dst_root = HF_LEROBOT_HOME / dst_repo_id
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(f"{dst_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(dst_root)

    info = json.loads((src_root / "meta/info.json").read_text())
    chunks_size = info["chunks_size"]
    data_path_tmpl = info["data_path"]
    if info.get("total_videos", 0):
        raise NotImplementedError(
            f"{src_repo_id} stores frames as video; this script only copies parquet-embedded images"
        )

    episodes = [
        json.loads(line) for line in (src_root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()
    ]
    if len(episodes) != info["total_episodes"]:
        raise RuntimeError(f"episodes.jsonl has {len(episodes)} rows, info says {info['total_episodes']}")

    # v2.1 keeps per-episode stats alongside; v2.0 (the released LIBERO) has a single
    # aggregate meta/stats.json instead.
    stats_path = src_root / "meta/episodes_stats.jsonl"
    episode_stats = None
    if stats_path.exists():
        episode_stats = {
            json.loads(line)["episode_index"]: json.loads(line)
            for line in stats_path.read_text().splitlines()
            if line.strip()
        }

    print(f"Source: {src_repo_id} ({len(episodes)} episodes, {info['total_frames']} frames)")
    selected = _select_episodes(episodes, scale, seed)
    print(f"Subset: {dst_repo_id} ({len(selected)} episodes, scale={scale}, seed={seed})")

    new_episodes = []
    new_stats = []
    global_idx = 0
    for new_ep_idx, src_ep_idx in enumerate(tqdm.tqdm(selected, desc="copying episodes")):
        src_parquet = _episode_parquet_path(src_root, data_path_tmpl, src_ep_idx, chunks_size)
        dst_parquet = _episode_parquet_path(dst_root, data_path_tmpl, new_ep_idx, chunks_size)
        dst_parquet.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(src_parquet)
        n = table.num_rows

        # episode_index is constant per episode and `index` is the running frame counter
        # over the whole dataset, so both are renumbered; frame_index is per-episode and
        # task_index still points into the (unchanged) tasks.jsonl.
        cols = []
        for name in table.column_names:
            if name == "episode_index":
                cols.append(pa.array([new_ep_idx] * n, type=pa.int64()))
            elif name == "index":
                cols.append(pa.array(range(global_idx, global_idx + n), type=pa.int64()))
            else:
                cols.append(table.column(name))
        new_table = pa.table(cols, names=table.column_names).replace_schema_metadata(table.schema.metadata)
        pq.write_table(new_table, dst_parquet)

        new_episodes.append({**episodes[src_ep_idx], "episode_index": new_ep_idx})
        if episode_stats is not None:
            new_stats.append({**episode_stats[src_ep_idx], "episode_index": new_ep_idx})
        global_idx += n

    meta_dst = dst_root / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)
    (meta_dst / "info.json").write_text(
        json.dumps(
            {
                **info,
                "total_episodes": len(selected),
                "total_frames": global_idx,
                "total_chunks": (len(selected) - 1) // chunks_size + 1,
                "splits": {"train": f"0:{len(selected)}"},
            },
            indent=4,
        )
    )
    with (meta_dst / "episodes.jsonl").open("w") as f:
        for e in new_episodes:
            f.write(json.dumps(e) + "\n")
    if episode_stats is not None:
        with (meta_dst / "episodes_stats.jsonl").open("w") as f:
            for s in new_stats:
                f.write(json.dumps(s) + "\n")
    shutil.copy(src_root / "meta/tasks.jsonl", meta_dst / "tasks.jsonl")
    if (src_root / "meta/stats.json").exists():
        # Aggregate stats over the *full* dataset. openpi normalizes from its own
        # assets/<config>/<repo_id>/norm_stats.json, computed per config by
        # scripts/compute_norm_stats.py, so this file is only here to keep the dataset
        # loadable by LeRobot's v2.0 metadata reader.
        shutil.copy(src_root / "meta/stats.json", meta_dst / "stats.json")

    print(f"Wrote subset to: {dst_root}")
    print(f"  total_episodes: {len(selected)}")
    print(f"  total_frames:   {global_idx}")


if __name__ == "__main__":
    tyro.cli(main)
