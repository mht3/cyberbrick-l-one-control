#!/usr/bin/env python3
"""lerobot-train with this repo's defaults, a held-out set, and action balancing.

    python scripts/train.py --policy.type=act --dataset.repo_id=lone/l_one_train ...

Takes exactly the arguments lerobot-train takes and forwards them untouched -- see
the README for the pi0.5 invocation. It only sets defaults that would otherwise
have to be repeated on every run, and adds two things lerobot has no equivalent
for:

    --balance-actions
    --eval-dataset.repo-id / --eval-dataset.root

lerobot's own held-out split (`--dataset.eval_split`) carves episodes out of the
training dataset. That is not what we want here: demonstrations recorded in one
session share lighting, start pose and marker placement, so episodes held out of
the same folder measure memorization far more than generalization. Pointing eval
at a separately collected folder makes the eval loss mean what it should. The two
datasets must share a schema -- same fps, same camera views, same action space --
since the policy is scored with the preprocessor it trained with.

`--balance-actions` evens out what the policy is shown. Teleop produces a badly
skewed diet -- on the first L-ONE dataset the base motor moves in 9.4% of frames
while nothing moves at all in 48.9% -- and a policy trained on it can score well
by learning the marginals rather than the task. Every frame is sorted into one of
ACTION_SCENARIOS and the training sampler draws the five equally often, so the
dataset stays a faithful recording while the batches do not inherit its skew.

Balancing by sampling rather than by deleting frames is deliberate. Idle frames
are real supervision -- holding still is part of the task -- and a chunked policy
trains on runs of consecutive actions, so cutting frames out of an episode leaves
chunks that span stretched, discontinuous time and teach the motion too fast.

WHAT IT BALANCES, EXACTLY
-------------------------
The sampler chooses which frame a training chunk *starts* at, so what is evened
out is the distribution of chunk heads -- the observation the policy conditions
on and the first action it commits to. It does not reweight the actions inside a
chunk, and it cannot: a 50-step chunk spans two seconds and mostly contains
whatever is common, whichever frame it starts from.

Measured on l_one_manipulation_rebalanced_train with chunk_size 50 and
n_action_steps 10, against a natural base-motion rate of 9.3%:

    chunk start frames    evened to ~20% per scenario
    executed window (10)  14.6% base motion   1.57x
    whole chunk (50)      10.5% base motion   1.13x

The executed window is the part that reaches the arm before the next inference,
so that is the number that matters; the tail of each chunk is largely unchanged.
Do not read "five scenarios at 20%" as a claim about the loss, which averages
over all 50 steps.

Chunking itself is untouched -- every sampled chunk is still the consecutive
actions from its start frame, inside one episode, so this works identically for
flow-matching (pi0.5) and for L1/MSE chunk regression (ACT).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lone_data  # noqa: F401  -- sets HF_HUB_OFFLINE before lerobot is imported

# The PaliGemma tokenizer is built once in the main process, then --num_workers forks
# the dataloader, and HF's Rust tokenizer warns once per fork. Nothing tokenizes inside
# a worker, so the parallel path buys nothing. Must precede any tokenizers import.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# lone_data disables Hub access for every entry point in the repo. Training is the one
# that legitimately needs it, for pi05_base and the gated PaliGemma tokenizer, so undo
# it here. LONE_TRAIN_OFFLINE=1 keeps the offline default -- useful for ACT, which needs
# no Hub access at all.
if os.environ.get("LONE_TRAIN_OFFLINE") != "1":
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

# How often to score the held-out set when one is given and --eval_steps is not.
DEFAULT_EVAL_STEPS = 5000

# The five things the arm can be doing in a frame, most specific first. A frame is
# assigned to the first one that matches, so a frame driving both the base and the
# lower arm counts as base motion -- the scarcer signal wins the frame rather than
# being buried in the commoner one.
#
# Dimensions 0-2 are speeds, so "moving" means not at the stop level. Dimension 3
# is a position latch rather than a speed, so what it contributes is a state: the
# gripper closed with nothing else happening is the arm holding an object, which
# is a distinct thing to recognize from the empty-handed pause that follows it.
ACTION_SCENARIOS = ("base moving", "upper arm moving", "lower arm moving",
                    "gripper closed (holding)", "no action")


def take_flag(name):
    """Pop `--name=value` (or `--name value`) out of sys.argv and return the value.

    Done by hand rather than with argparse because everything else on the command
    line belongs to draccus, which rejects any flag it does not recognise -- so
    these have to be gone before it ever sees argv.
    """
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            del sys.argv[i + 1:i + 3]
            return args[i + 1]
        if arg.startswith(name + "="):
            del sys.argv[i + 1]
            return arg.split("=", 1)[1]
    return None


def take_switch(name):
    """Pop a bare `--name` out of sys.argv, returning whether it was there."""
    if name in sys.argv[1:]:
        sys.argv.remove(name)
        return True
    return False


def has_flag(name):
    return any(a == name or a.startswith(name + "=") for a in sys.argv[1:])


EVAL_REPO_ID = take_flag("--eval-dataset.repo-id")
EVAL_ROOT = take_flag("--eval-dataset.root")
BALANCE_ACTIONS = take_switch("--balance-actions")

from lerobot.scripts import lerobot_train  # noqa: E402
from lerobot.configs.train import TrainPipelineConfig  # noqa: E402

# push_to_hub defaults to True, so TrainPipelineConfig.validate() refuses to start
# without a policy.repo_id to push the trained model to. An explicit
# --policy.push_to_hub=true still wins. This concerns the checkpoint, not the dataset.
if not has_flag("--policy.push_to_hub"):
    sys.argv.append("--policy.push_to_hub=false")


def _eval_dataset(cfg, train_dataset):
    """The held-out dataset, built the way lerobot builds its own eval split.

    Same delta_timestamps, same uint8 return and tolerance as the training set, and
    no image transforms -- an augmented eval set would score a different input
    distribution every time it ran. delta_timestamps come from the *training*
    metadata so both datasets are sliced identically even if the eval folder's
    metadata differs in some way it should not.
    """
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import IMAGENET_STATS

    dataset = LeRobotDataset(
        EVAL_REPO_ID or cfg.dataset.repo_id,
        root=EVAL_ROOT,
        delta_timestamps=resolve_delta_timestamps(cfg.trainable_config, train_dataset.meta),
        image_transforms=None,
        video_backend=cfg.dataset.video_backend,
        return_uint8=True,
        tolerance_s=cfg.tolerance_s,
    )
    if cfg.dataset.use_imagenet_stats:
        for key in dataset.meta.camera_keys:
            for stats_type, stats in IMAGENET_STATS.items():
                dataset.meta.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)
    return dataset


def scenario_of(actions):
    """Which of ACTION_SCENARIOS each frame belongs to, as indices into it.

    `actions` is (N, 4) in raw command units, in dataset order.
    """
    import numpy as np

    from lone_data.features import ACTION_LEVELS

    def moving(dim):
        levels = np.array(ACTION_LEVELS[dim])
        # Index 1 is the stop level on every speed channel; anything else is motion.
        return np.argmin(np.abs(actions[:, dim, None] - levels[None, :]), axis=1) != 1

    base, upper, lower = moving(0), moving(1), moving(2)
    still = ~(base | upper | lower)
    # Midway between the open and closed angles, so this does not depend on the
    # exact values teleop happened to emit.
    lo, hi = ACTION_LEVELS[3]
    closed = actions[:, 3] > (lo + hi) / 2

    out = np.full(len(actions), 4, dtype=np.int64)   # no action
    out[still & closed] = 3
    out[lower & ~(base | upper)] = 2
    out[upper & ~base] = 1
    out[base] = 0
    return out


def dataset_actions(dataset):
    """Every action in a dataset, in dataset-index order, read straight from parquet.

    Going through dataset[i] would decode a video frame per action -- minutes of
    work to read a column that is already sitting in the parquet next to it.
    """
    import glob

    import numpy as np
    import pyarrow.parquet as pq

    from lerobot.utils.constants import ACTION

    files = sorted(glob.glob(os.path.join(str(dataset.root), "data", "**", "*.parquet"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {dataset.root}/data")
    tables = [pq.read_table(f, columns=["episode_index", "frame_index", ACTION]) for f in files]
    episode = np.concatenate([np.asarray(t["episode_index"]) for t in tables])
    frame = np.concatenate([np.asarray(t["frame_index"]) for t in tables])
    actions = np.concatenate([np.asarray(t[ACTION].to_pylist(), dtype=np.float32)
                              for t in tables])
    # Dataset index order is (episode_index, frame_index); the files are not
    # required to be in that order, and with 11 of them they are not.
    return actions[np.lexsort((frame, episode))]


class BalancedSampler:
    """EpisodeAwareSampler's index pool, drawn so every scenario appears equally.

    Same interface as the sampler it replaces -- iterating yields dataset indices,
    `__len__` is an epoch, and state_dict/load_state_dict resume it -- because
    lerobot's training loop drives all three and computes its resume offset from
    the length.

    Draws are with replacement, which is what makes the weighting mean anything:
    an epoch is the same number of samples as before, with scarce scenarios
    repeated and abundant ones passed over. Like the sampler it wraps, the order
    is a pure function of (seed, epoch), so every rank produces the same
    permutation without sharing an RNG and a resume lands on the same sample.
    """

    def __init__(self, base, scenarios, seed=0):
        import numpy as np

        self._pool = np.asarray(base.indices, dtype=np.int64)
        self._num_frames = len(self._pool)
        self.seed = seed
        self._epoch = 0
        self._start_index = 0

        labels = scenarios[self._pool]
        counts = np.bincount(labels, minlength=len(ACTION_SCENARIOS)).astype(np.float64)
        # A scenario absent from this dataset gets no weight rather than an
        # infinite one; the rest still come out equal among themselves.
        share = np.divide(1.0, counts, out=np.zeros_like(counts), where=counts > 0)
        self._weights = share[labels]
        self._weights /= self._weights.sum()
        self.counts = counts

    def report(self):
        total = self.counts.sum()
        lines = [f"[train] balancing {int(total)} frames across {len(ACTION_SCENARIOS)} scenarios:"]
        for name, n in zip(ACTION_SCENARIOS, self.counts):
            if not n:
                lines.append(f"          {name:26s}      0  absent -- not sampled")
                continue
            factor = (total / len(ACTION_SCENARIOS)) / n
            lines.append(f"          {name:26s} {int(n):6d}  {100 * n / total:5.1f}%"
                         f"  -> x{factor:.2f}")
        return "\n".join(lines)

    def set_epoch(self, epoch):
        self._epoch = epoch

    def state_dict(self):
        return {"epoch": self._epoch, "start_index": self._start_index}

    def load_state_dict(self, state):
        self._epoch = state["epoch"]
        self._start_index = state["start_index"]

    def __iter__(self):
        import numpy as np

        epoch, start = self._epoch, self._start_index
        self._epoch += 1
        self._start_index = 0
        seed = int(np.random.SeedSequence([self.seed, epoch]).generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(seed)
        draw = rng.choice(self._pool, size=self._num_frames, replace=True, p=self._weights)
        return iter(draw[start:].tolist())

    def __len__(self):
        return self._num_frames


def _install_balanced_sampler(dataset):
    """Point lerobot's training loop at a scenario-balanced sampler."""
    import numpy as np

    original = lerobot_train.EpisodeAwareSampler

    def build(*a, **kw):
        base = original(*a, **kw)
        try:
            scenarios = scenario_of(dataset_actions(dataset))
            if len(scenarios) < int(np.max(base.indices)) + 1:
                raise ValueError(
                    f"read {len(scenarios)} actions but the sampler indexes up to "
                    f"{int(np.max(base.indices))}"
                )
            sampler = BalancedSampler(base, scenarios, seed=kw.get("seed", 0))
        except Exception as e:
            # Never take a training run down over this: an unbalanced run is a
            # worse run, not a failed one, and the message says which happened.
            print(f"[train] --balance-actions could not be applied ({e}); "
                  "falling back to the standard sampler.")
            return base
        print(sampler.report(), flush=True)
        return sampler

    lerobot_train.EpisodeAwareSampler = build


def _make_train_eval_datasets(cfg):
    """Replaces lerobot's splitter: train is the whole training folder, eval is ours."""
    from lerobot.datasets.factory import make_dataset

    train_dataset = make_dataset(cfg)
    eval_dataset = _eval_dataset(cfg, train_dataset) if (EVAL_ROOT or EVAL_REPO_ID) else None
    if eval_dataset is not None:
        print(f"[train] held-out eval set: {EVAL_ROOT} "
              f"({eval_dataset.num_episodes} episode(s), {len(eval_dataset)} frames)")
    if BALANCE_ACTIONS:
        _install_balanced_sampler(train_dataset)
    return train_dataset, eval_dataset


def _install_splitter():
    """Route dataset construction through ours.

    It is the one place that has the built training dataset in hand, which is what
    both features need: the eval folder is attached beside it, and the balanced
    sampler has to read its actions.
    """
    lerobot_train.make_train_eval_datasets = _make_train_eval_datasets


def _allow_eval_without_split():
    """Let eval_steps > 0 through with dataset.eval_split at 0.0.

    lerobot only knows about the in-dataset split and validate() refuses that
    combination, so eval_split is nudged non-zero for the duration of that call
    alone and restored afterwards -- the replaced splitter never reads it, and
    train_config.json keeps recording the 0.0 that is actually true.
    """
    original_validate = TrainPipelineConfig.validate

    def validate(self):
        saved = self.dataset.eval_split
        self.dataset.eval_split = saved or 1e-6
        try:
            original_validate(self)
        finally:
            self.dataset.eval_split = saved

    TrainPipelineConfig.validate = validate


if EVAL_ROOT or EVAL_REPO_ID or BALANCE_ACTIONS:
    _install_splitter()

if EVAL_ROOT or EVAL_REPO_ID:
    if not EVAL_ROOT:
        from lone_data.checkpoints import default_root

        EVAL_ROOT = default_root(EVAL_REPO_ID)
    _allow_eval_without_split()
    if not has_flag("--eval_steps"):
        sys.argv.append(f"--eval_steps={DEFAULT_EVAL_STEPS}")

if __name__ == "__main__":
    lerobot_train.main()
