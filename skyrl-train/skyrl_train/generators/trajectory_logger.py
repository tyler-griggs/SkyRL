from typing import Any, Dict, List, Optional, Tuple
import json
import random

from skyrl_train.generators.base import TrajectoryID, TrainingPhase
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.utils.tracking import Tracking


# TODO(tgriggs): Note that this API will change and be migrated to the central buffer.
# Will introduce Trajectory and TrajectoryGroup classes to further simplify this API.
# Create super simple base class --> then add Experimental
class TrajectoryLogger:
    """
    Minimal trajectory logging helper. Groups rows by instance_id, samples groups,
    and uses the provided Tracking instance to emit either a wandb.Table (if backend
    is W&B) or a JSON payload. Associates every log with a step and phase.
    """

    def __init__(self, tracker: Tracking, sample_rate: float = 0.05):
        self.tracker = tracker
        self.sample_rate = sample_rate

    def log_batch(
        self,
        *,
        trajectory_ids: List[Optional[TrajectoryID]],
        prompts: List[ConversationType],
        responses: List[ConversationType],
        rewards: List[float],
        model_version: int,
        phase: TrainingPhase,
    ) -> None:
        if not (self.sample_rate > 0 and self.tracker):
            return

        # Bucket rows by instance_id.
        group_id_to_indices = {}
        for i, traj_id in enumerate(trajectory_ids):
            group_id_to_indices.setdefault(traj_id.instance_id, []).append(i)

        # Determine which groups to log.
        selected_groups = []
        for group_id, idxs in group_id_to_indices.items():
            if not self._should_log():
                continue
            for i in idxs:
                prompt_txt = self._conversation_to_json(prompts[i])
                convo_txt = self._conversation_to_json(responses[i])
                selected_groups.append(
                    (
                        group_id,
                        trajectory_ids[i].repetition_id,
                        prompt_txt,
                        convo_txt,
                        rewards[i],
                    )
                )
        if not selected_groups:
            return

        # Build trajectory log object (wandb table or json dict)
        if self._is_wandb():
            payload = self._to_wandb_table(selected_groups)
        else:
            payload = self._to_json(selected_groups)

        # Log to tracker
        self._tracker_log(payload, model_version, phase)

    # ---------- internals ----------

    def _should_log(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        return random.random() < self.sample_rate

    def _conversation_to_json(self, msgs: ConversationType) -> str:
        # Convert conversation to JSON for readability. Each message is {"role": "...", "content": "..."}.
        return json.dumps(list(msgs), ensure_ascii=False, separators=(",", ":"))

    def _is_wandb(self) -> bool:
        # Check if tracker has wandb backend
        logger_dict = getattr(self.tracker, "logger", None)
        if isinstance(logger_dict, dict):
            return "wandb" in logger_dict
        return False

    def _tracker_log(self, payload: Dict[str, Any], model_version: int, phase: TrainingPhase) -> None:
        # TODO(tgriggs): Add phase to payload
        self.tracker.log(payload, step=model_version)

    def _to_wandb_table(
        self,
        groups: List[Tuple[str, int, str, str, float]],
    ) -> Dict[str, Any]:
        import wandb

        table = wandb.Table(columns=["group_id", "repetition_id", "prompt", "conversation", "reward"])
        for r in groups:
            table.add_data(*r)
        return {"trajectories": table}

    def _to_json(
        self,
        rows: List[Tuple[str, int, str, str, float]],
    ) -> Dict[str, Any]:
        return {
            "trajectories": [
                {
                    "group_id": gid,
                    "repetition_id": rid,
                    "prompt": prompt,
                    "conversation": conv,
                    "reward": reward,
                }
                for gid, rid, prompt, conv, reward in rows
            ]
        }
