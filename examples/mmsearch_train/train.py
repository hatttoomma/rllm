import hydra

from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer

from .workflow import MMSearchWorkflow

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):

    trainer = AgentTrainer(
        workflow_class=MMSearchWorkflow,
        workflow_args={},
        config=config,
    )
    trainer.train()


if __name__ == "__main__":
    main()
