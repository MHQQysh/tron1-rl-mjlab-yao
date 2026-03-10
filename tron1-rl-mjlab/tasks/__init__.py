from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabOnPolicyRunner

from .cfg.wf_tron_env_cfg import make_wf_tron_env_cfg
from .cfg.wf_tron_rl_cfg import WfTronRlCfg

register_mjlab_task(
    task_id="Mjlab-WF-Tron",
    env_cfg=make_wf_tron_env_cfg(),
    play_env_cfg=make_wf_tron_env_cfg(),
    rl_cfg=WfTronRlCfg(),
    runner_cls=MjlabOnPolicyRunner,
)
