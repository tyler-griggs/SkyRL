from typing import Any, Dict, List, Optional, Tuple
import json
import random

from skyrl_train.generators.base import TrajectoryID, TrainingPhase
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.utils.tracking import Tracking


class TrajectoryLogger:
    """
    Logs trajectories.

    Warning:
        This API is experimental and will likely change soon as we refactor how
        Trajectory and TrajectoryGroup objects are handled. We include this code
        to provide helpful functionality now, but expect the interface to change
        in future versions.
    """

    def __init__(self, tracker: Tracking, sample_rate: float = 0.05):
        self.tracker = tracker
        self.sample_rate = sample_rate

    def log(
        self,
        trajectory_ids: List[Optional[TrajectoryID]],
        prompts: List[ConversationType],
        responses: List[ConversationType],
        rewards: List[float],
        model_version: int,
        # TODO(tgriggs): Add training phase to log.
        phase: TrainingPhase,
    ) -> None:
        if self.sample_rate <= 0 or not self.tracker:
            return

        # Form trajectories groupings by instance_id.
        group_id_to_indices = {}
        for i, traj_id in enumerate(trajectory_ids):
            group_id_to_indices.setdefault(traj_id.instance_id, []).append(i)

        # Determine which groups to log.
        selected_groups = []
        for group_id, idxs in group_id_to_indices.items():
            if self._should_log():
                selected_groups.extend(
                    (
                        group_id,
                        trajectory_ids[i].repetition_id,
                        self._conversation_to_json(prompts[i]),
                        self._conversation_to_json(responses[i]),
                        rewards[i],
                    )
                    for i in idxs
                )
        if not selected_groups:
            return

        # Log to tracker.
        if self._is_wandb():
            self.tracker.log_to_backend("wandb", self._to_wandb_table(selected_groups), model_version, commit=True)
        else:
            self.tracker.log(self._to_json(selected_groups), model_version)

    def _should_log(self) -> bool:
        return self.sample_rate >= 1.0 or random.random() < self.sample_rate

    def _conversation_to_json(self, msgs: ConversationType) -> str:
        # Convert conversation to JSON for readability. Each message is {"role": "...", "content": "..."}.
        return json.dumps(list(msgs), ensure_ascii=False, separators=(",", ":"))

    def _is_wandb(self) -> bool:
        return "wandb" in self.tracker.logger

    def _to_wandb_table(
        self,
        groups: List[Tuple[str, int, str, str, float]],
    ) -> Dict[str, Any]:
        import wandb

        table = wandb.Table(columns=["group_id", "repetition_id", "prompt", "conversation", "reward"])
        for group in groups:
            table.add_data(*group)
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
