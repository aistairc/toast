import collections
import dataclasses
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Sharding: split the suite's tasks across several processes so one suite can be
    # evaluated on several GPUs at once. Shard i takes tasks i, i + num_shards, ... Each
    # shard talks to its own policy server; sum the per-shard results (--args.results-out-path) afterwards.
    #################################################################################################################
    num_shards: int = 1
    shard_index: int = 0

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    save_videos: bool = True  # Encoding a replay video per episode costs ~10% of the rollout time

    # Where to write the per-task success counts (JSON). Unlike the log lines, this is
    # machine-readable and shard-aware. It is rewritten after every task, so it doubles as a
    # checkpoint: a re-run with the same arguments picks up at the first unfinished task
    # (see --args.no-resume to force a clean run).
    results_out_path: str | None = None
    resume: bool = True

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.save_videos:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"shard_index must be in [0, {args.num_shards}), got {args.shard_index}")
    task_ids = list(range(args.shard_index, num_tasks_in_suite, args.num_shards))
    if args.num_shards > 1:
        logging.info(f"Shard {args.shard_index}/{args.num_shards}: tasks {task_ids} of {num_tasks_in_suite}")

    # Start evaluation. The results JSON doubles as a checkpoint -- a rollout that dies
    # inside the simulator takes the whole process with it and leaves no traceback, so a
    # re-run picks up at the episode after the last one recorded rather than starting over.
    # Episode k always replays initial state k, so resuming neither repeats nor skips one.
    recorded = _load_recorded_tasks(args)
    per_task_results = [recorded[t] for t in task_ids if t in recorded]
    total_episodes = sum(t["episodes"] for t in per_task_results)
    total_successes = sum(t["successes"] for t in per_task_results)
    if total_episodes:
        logging.info(f"Resuming: {total_episodes} episode(s) already recorded")

    for task_id in tqdm.tqdm(task_ids):
        record = recorded.get(task_id)
        task_episodes = record["episodes"] if record is not None else 0
        task_successes = record["successes"] if record is not None else 0
        if task_episodes >= args.num_trials_per_task:
            continue

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        if record is None:
            record = {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
            }
            per_task_results.append(record)

        # Start episodes
        for episode_idx in tqdm.tqdm(range(task_episodes, args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            if args.save_videos:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            # Checkpoint before logging: the next episode is where this process is most
            # likely to die, and the record is what a retry resumes from.
            record["episodes"] = task_episodes
            record["successes"] = task_successes
            _write_results(args, per_task_results)

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    if total_episodes:
        logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    _write_results(args, per_task_results)
    if args.results_out_path is not None:
        logging.info(f"Wrote results to {args.results_out_path}")


def _results_payload(args: Args, per_task_results: list[dict]) -> dict:
    return {
        "task_suite_name": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "total_episodes": sum(t["episodes"] for t in per_task_results),
        "total_successes": sum(t["successes"] for t in per_task_results),
        "tasks": per_task_results,
    }


def _write_results(args: Args, per_task_results: list[dict]) -> None:
    """Rewrite the results JSON. Written via a temp file: the process can die at any moment
    (a simulator abort takes it down with no traceback) and a half-written checkpoint would
    be worse than none."""
    if args.results_out_path is None:
        return
    out_path = pathlib.Path(args.results_out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_results_payload(args, per_task_results), indent=2))
    tmp_path.replace(out_path)


def _load_recorded_tasks(args: Args) -> dict[int, dict]:
    """Per-task results of an earlier run of this same shard, keyed by task id.

    Partly evaluated tasks are included: the checkpoint is written after every episode, so
    a task with ``episodes < num_trials_per_task`` resumes at ``episodes``.
    """
    if not args.resume or args.results_out_path is None:
        return {}
    out_path = pathlib.Path(args.results_out_path)
    if not out_path.exists():
        return {}
    try:
        payload = json.loads(out_path.read_text())
    except json.JSONDecodeError:
        logging.warning(f"{out_path} is not readable JSON; starting over")
        return {}

    settings = ("task_suite_name", "num_trials_per_task", "num_shards", "shard_index", "seed", "replan_steps")
    expected = _results_payload(args, [])
    if any(payload.get(k) != expected[k] for k in settings):
        logging.warning(f"{out_path} was written with different settings; starting over")
        return {}
    return {
        t["task_id"]: t
        for t in payload.get("tasks", [])
        if 0 <= t.get("successes", -1) <= t.get("episodes", -1) <= args.num_trials_per_task
    }


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
