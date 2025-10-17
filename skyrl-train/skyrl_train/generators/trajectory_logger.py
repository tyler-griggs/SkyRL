from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import hashlib
import json

from skyrl_train.utils.tracking import Tracking

@dataclass
class TrajectoryLoggerConfig:
    enabled: bool = False
    # Fraction of GRPO groups (by instance_id) to log. 1.0 -> log all groups.
    sample_frac: float = 0.05
    # Name used when logging the table
    table_name: str = "trajectories"
    # Optional explicit backend (e.g. "wandb"); if None, infer best-effort
    backend: Optional[str] = None

class TrajectoryLogger:
    """
    Minimal trajectory logging helper. Groups rows by instance_id, samples groups,
    and uses the provided Tracking to emit either a wandb.Table (if backend is W&B)
    or a JSON payload.
    """
    def __init__(self, tracker: Tracking, cfg: TrajectoryLoggerConfig):
        self.tracker = tracker
        self.cfg = cfg

    # ---------- public API ----------

    def log_batch(
        self,
        *,
        group_ids: Sequence[str],            # instance_id per trajectory
        repetition_ids: Sequence[int],       # repetition_id per trajectory
        init_prompts: Sequence[Sequence[Dict[str, str]]],  # list of role/content dicts
        conversations: Sequence[Sequence[Dict[str, str]]], # list of role/content dicts
        final_rewards: Sequence[float],
        global_step: Optional[int] = None,
    ) -> None:
        if not (self.cfg.enabled and self.tracker):
            return

        # Bucket rows by GRPO group (instance_id).
        buckets: Dict[str, List[int]] = {}
        for i, gid in enumerate(group_ids):
            buckets.setdefault(gid, []).append(i)

        # Decide which groups to log (deterministic hash sampling).
        selected_rows: List[Tuple[str, int, str, str, float]] = []
        for gid, idxs in buckets.items():
            if not self._should_log_group(gid):
                continue
            for i in idxs:
                prompt_txt = self._messages_to_json(init_prompts[i])
                convo_txt  = self._messages_to_json(conversations[i])
                # row: (group_id, repetition_id, prompt, conversation, final_reward)
                selected_rows.append((
                    gid,
                    int(repetition_ids[i]),
                    prompt_txt,
                    convo_txt,
                    float(final_rewards[i]),
                ))

        if not selected_rows:
            return

        # Emit
        if self._is_wandb():
            self._emit_wandb(selected_rows, global_step)
        else:
            self._emit_json(selected_rows, global_step)

    # ---------- internals ----------

    def _should_log_group(self, group_id: str) -> bool:
        if self.cfg.sample_frac >= 1.0:
            return True
        # Deterministic sampling by group_id
        h = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest(), 16)
        # Compare against fixed scale for stability across runs
        return (h % 10_000_000) < int(self.cfg.sample_frac * 10_000_000)

    def _messages_to_json(self, msgs: Sequence[Dict[str, str]]) -> str:
        # Keep the payload *very* simple and human-readable
        # Each message is {"role": "...", "content": "..."}
        return json.dumps(list(msgs), ensure_ascii=False, separators=(",", ":"))

    def _is_wandb(self) -> bool:
        if self.cfg.backend:
            return self.cfg.backend.lower() == "wandb"
        # Best-effort inference
        backend_name = (
            getattr(self.tracker, "backend", None)
            or getattr(self.tracker, "logger", None)
            or getattr(self.tracker, "name", None)
        )
        if isinstance(backend_name, str) and backend_name.lower() == "wandb":
            return True
        try:
            import wandb  # noqa: F401
            return True
        except Exception:
            return False

    def _emit_wandb(
        self,
        rows: Sequence[Tuple[str, int, str, str, float]],
        global_step: Optional[int],
    ) -> None:
        import wandb
        table = wandb.Table(
            columns=["group_id", "repetition_id", "prompt", "conversation", "final_reward"]
        )
        for r in rows:
            table.add_data(*r)

        payload = {self.cfg.table_name: table}
        # Use common Tracker interfaces without assuming implementation details
        if hasattr(self.tracker, "log"):
            self.tracker.log(payload, step=global_step)
        elif hasattr(self.tracker, "wandb_run"):
            # If tracker exposes the underlying run
            self.tracker.wandb_run.log(payload, step=global_step)
        else:
            # Fallback if tracker is thin and wandb run is context-managed globally
            wandb.log(payload, step=global_step)

    def _emit_json(
        self,
        rows: Sequence[Tuple[str, int, str, str, float]],
        global_step: Optional[int],
    ) -> None:
        payload = {
            self.cfg.table_name: [
                {
                    "group_id": gid,
                    "repetition_id": rid,
                    "prompt": prompt,
                    "conversation": conv,
                    "final_reward": reward,
                }
                for gid, rid, prompt, conv, reward in rows
            ]
        }
        if hasattr(self.tracker, "log"):
            self.tracker.log(payload, step=global_step)
