from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def WfTronRlCfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for WF-TRON task."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=15000,
        save_interval=200,
        experiment_name="wf_tron_1a_flat",
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )
