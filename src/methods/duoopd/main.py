"""Train DuoOPD, or its sampled-token OPD and ablation variants, on native verl."""

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.ray_trainer import Role
from verl.utils import hf_tokenizer

from .objective import resolve_switches
from .teacher import TeacherWorker
from .trainer import DuoOPDTrainer


class DuoOPDTaskRunner(TaskRunner):
    trainer_class = DuoOPDTrainer

    def add_reward_model_worker(self, config):
        super().add_reward_model_worker(config)
        switches = config.algorithm.duoopd
        with open_dict(config.reward_model), open_dict(config.reward_model.reference):
            # The teacher cache is needed for the cell split (online mean) and the reference.
            config.reward_model.load_teacher_cache = bool(
                switches.online_mean or switches.response_mean
                or switches.teacher_reference or switches.opd_cell
            )
            config.reward_model.reference.use_teacher_answers = bool(
                switches.teacher_reference and switches.opd_cell != "teacher_only"
            )
        self.role_worker_mapping[Role.RewardModel] = ray.remote(TeacherWorker)

    def run(self, config):
        actor = config.actor_rollout_ref.actor
        rollout = config.actor_rollout_ref.rollout
        if config.algorithm.adv_estimator != "duoopd":
            raise ValueError("Set algorithm.adv_estimator=duoopd; variants are selected by algorithm.duoopd.*")
        with open_dict(config.algorithm):
            config.algorithm.duoopd = OmegaConf.create(resolve_switches(config.algorithm))
        if (actor.strategy != "fsdp" or config.reward_model.strategy != "fsdp"
                or config.trainer.use_legacy_worker_impl != "enable"
                or not config.reward_model.enable or config.reward_model.use_reward_loop):
            raise ValueError("Use native legacy FSDP actor and the colocated teacher worker")
        if (config.data.shuffle or config.trainer.balance_batch or actor.shuffle
                or actor.ppo_epochs != 1 or actor.ppo_mini_batch_size != config.data.train_batch_size
                or actor.use_dynamic_bsz or actor.loss_agg_mode != "seq-mean-token-mean" or rollout.n != 1):
            raise ValueError("Preserve fixed prompt batches, fixed microbatches and one response-mean update")
        if (config.algorithm.use_kl_in_reward or actor.use_kl_loss or actor.entropy_coeff != 0
                or actor.policy_loss.loss_mode != "vanilla"
                or config.algorithm.rollout_correction is not None):
            raise ValueError("The frozen objective uses PPO clipping without extra KL, entropy or rollout correction")
        if config.reward_model.launch_reward_fn_async or config.reward_model.enable_resource_pool:
            raise ValueError("Teacher scoring uses the same GPU pool before each actor update")
        if config.reward_model.model.input_tokenizer is not None:
            raise ValueError("Teacher scoring must preserve actual student token IDs")
        student = hf_tokenizer(config.actor_rollout_ref.model.path)
        teacher = hf_tokenizer(config.reward_model.model.path)
        if student.get_vocab() != teacher.get_vocab() or student.eos_token_id != teacher.eos_token_id:
            raise ValueError("Teacher and student must share token IDs and EOS semantics")
        return super().run(config)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config):
    from methods.duoopd.main import DuoOPDTaskRunner as Runner

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(Runner))


if __name__ == "__main__":
    main()
