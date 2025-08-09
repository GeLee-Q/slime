import atexit
import os
import re
import time
from typing import Any, Dict, Optional

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore


_writer: Optional[SummaryWriter] = None

# Maintain recent step values for namespaces
_current_steps: Dict[str, int] = {
    "train": 0,
    "rollout": 0,
    "eval": 0,
    "perf": 0,
}

# Map metric prefixes to their step namespaces (mirrors define_metric in wandb_utils)
_prefix_to_namespace = {
    "train/": "train",
    "rollout/": "rollout",
    "multi_turn/": "rollout",
    "passrate/": "rollout",
    "eval/": "eval",
    "perf/": "rollout",
}


def _default_logdir(args) -> str:
    # runs/<project>/<group>/<timestamp>
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    project = getattr(args, "wandb_project", None) or "slime"
    group = getattr(args, "wandb_group", None) or "default"
    # Sanitize to avoid path issues
    def sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name)

    project = sanitize(project)
    group = sanitize(group)
    return os.path.join("runs", project, group, timestamp)


def init_tensorboard(args) -> Optional[str]:
    global _writer
    if not getattr(args, "use_tensorboard", False):
        return None

    if _writer is not None:
        return None

    logdir = getattr(args, "tensorboard_logdir", None) or _default_logdir(args)
    os.makedirs(logdir, exist_ok=True)
    if SummaryWriter is None:
        return None
    _writer = SummaryWriter(log_dir=logdir)

    # ensure writer is closed at exit
    atexit.register(close_tensorboard)
    return logdir


def init_tensorboard_secondary(args) -> None:
    # For TensorBoard, secondary processes can also write; each gets its own event file.
    init_tensorboard(args)


def get_writer() -> Optional[SummaryWriter]:
    return _writer


def close_tensorboard():
    global _writer
    try:
        if _writer is not None:
            _writer.flush()
            _writer.close()
    finally:
        _writer = None


def _update_current_steps_from_log_dict(log_dict: Dict[str, Any]):
    # Capture explicit step updates so later metrics without explicit step can use them
    for ns in ("train", "rollout", "eval", "perf"):
        key = f"{ns}/step"
        if key in log_dict:
            try:
                _current_steps[ns] = int(log_dict[key])
            except Exception:
                pass


def _infer_namespace(tag: str) -> Optional[str]:
    for prefix, ns in _prefix_to_namespace.items():
        if tag.startswith(prefix):
            return ns
    return None


def log_to_tensorboard(log_dict: Dict[str, Any]):
    writer = get_writer()
    if writer is None:
        return

    # First update known steps if provided in this dict
    _update_current_steps_from_log_dict(log_dict)

    for key, value in log_dict.items():
        # Skip step keys themselves
        if key.endswith("/step"):
            continue

        # Only log simple scalars
        if isinstance(value, (int, float)):
            namespace = _infer_namespace(key) or "rollout"
            step = _current_steps.get(namespace, 0)
            try:
                writer.add_scalar(key, float(value), global_step=step)
            except Exception:
                pass


