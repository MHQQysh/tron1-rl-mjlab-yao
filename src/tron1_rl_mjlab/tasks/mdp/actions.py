"""Custom action terms for the task."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseActionCfg, BaseAction
from mjlab.managers.action_manager import ActionTerm

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class JointVelocityAction(BaseAction):
    """Custom joint velocity action that uses write_ctrl directly."""

    cfg: JointVelocityActionCfg

    def __init__(self, cfg: JointVelocityActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        if cfg.use_default_offset:
            self._offset = torch.zeros_like(
                self._entity.data.default_joint_pos[:, self._target_ids]
            )

    def apply_actions(self):
        # Use write_ctrl to send control signals to actuators
        self._entity.data.write_ctrl(
            self._processed_actions, self._target_ids
        )


@dataclass(kw_only=True)
class JointVelocityActionCfg(BaseActionCfg):
    """Configuration for custom joint velocity action."""

    use_default_offset: bool = False

    def __post_init__(self):
        self.transmission_type = TransmissionType.JOINT

    def build(self, env: ManagerBasedRlEnv) -> JointVelocityAction:
        return JointVelocityAction(self, env)
