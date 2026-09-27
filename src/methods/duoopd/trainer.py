"""DuoOPD's full-batch feedback step; verl owns the training lifecycle."""

import numpy as np
import torch
import torch.nn.functional as F

from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .objective import duoopd_weights


class DuoOPDTrainer(RayPPOTrainer):
    def _compute_or_extract_reward(self, batch, reward_fn=None, return_dict=False, sum_reward=False):
        if "teacher_log_probs" in batch.batch:
            # AgentLoop supplies empty reward metadata; read teacher diagnostics
            # after union and return Python scalars for native JSONL logging.
            rewards = batch.batch["rm_scores"]
            extras = {key: values.tolist() for key, values in batch.non_tensor_batch.items()
                      if key.startswith("reference_")}
            if sum_reward:
                rewards = rewards.sum(-1)
            if return_dict:
                return {"reward_tensor": rewards, "reward_extra_info": extras}
            return rewards if sum_reward else (rewards, extras)
        return super()._compute_or_extract_reward(batch, reward_fn, return_dict, sum_reward)

    @torch.no_grad()
    def compute_advantage(self, batch):
        switches = self.config.algorithm.duoopd
        mask = batch.batch["response_mask"].bool()
        routed = batch.batch["teacher_log_probs"].float() - batch.batch["old_log_probs"].float()
        routed = routed.masked_fill(~mask, 0)
        student = batch.batch["rm_scores"].sum(-1)
        if not ((student == 0) | (student == 1)).all():
            raise ValueError("DuoOPD requires binary verifier outcomes")
        teacher = None
        if (switches.online_mean or switches.response_mean
                or switches.teacher_reference or switches.opd_cell):
            teacher = torch.as_tensor(batch.non_tensor_batch["reference_teacher_correct"], device=routed.device)
            if not ((teacher == 0) | (teacher == 1)).all():
                raise ValueError("DuoOPD requires binary cached teacher outcomes")
        task_ids = None
        if switches.online_mean_scope == "task":
            task_ids = torch.as_tensor(
                np.unique(batch.non_tensor_batch["data_source"], return_inverse=True)[1], device=routed.device
            )
        weights, constant = duoopd_weights(routed, mask, student, teacher, switches, task_ids)
        batch.batch["token_level_rewards"] = routed
        batch.batch["advantages"] = weights
        batch.batch["returns"] = weights.clone()
        if switches.outcome_cells:
            self._record_feedback(batch, weights, routed, mask, student, teacher, constant)
        return batch

    def _record_feedback(self, batch, weights, routed, mask, student, teacher, constant):
        """Per-answer feedback statistics for the native per-step rollout files.

        Replayed steps overwrite their files after recovery, so these means do not
        double-count replayed rows.
        """
        lengths = mask.sum(-1)
        # With the teacher reference on, routed values of teacher-correct/student-wrong
        # answers are dT, not d0.
        reference_cell = (teacher.bool() & ~student.bool()) if teacher is not None else torch.zeros_like(mask[:, 0])
        switches = self.config.algorithm.duoopd
        reference_on = bool(switches.teacher_reference and switches.opd_cell != "teacher_only")

        def answer_mean(values):
            return (values.masked_fill(~mask, 0).sum(-1) / lengths).cpu().tolist()

        diagnostics = {
            "task": batch.non_tensor_batch["data_source"].tolist(),
            "length": lengths.cpu().tolist(),
            "weight_mean": answer_mean(weights),
            "weight_abs_mean": answer_mean(weights.abs()),
            "weight_positive_fraction": answer_mean((weights > 0).float()),
            "d0_mean": answer_mean(routed),
            "d0_abs_mean": answer_mean(routed.abs()),
            "ordinary_softplus_mean": answer_mean(F.softplus(routed)),
            "batch_success_constant": [constant] * len(lengths),
            "reference_log_ratio_mean": answer_mean(routed),
        }
        if switches.online_mean_scope == "task" and switches.opd_cell != "student_only":
            diagnostics["task_success_constant"] = [
                value if not teacher_ok and student_ok else None
                for value, teacher_ok, student_ok in zip(
                    diagnostics["weight_mean"], teacher.bool().cpu().tolist(), student.bool().cpu().tolist(), strict=True
                )
            ]
        for row, is_reference in enumerate(reference_cell.cpu().tolist()):
            if is_reference and reference_on:
                for key in ("d0_mean", "d0_abs_mean", "ordinary_softplus_mean"):
                    diagnostics[key][row] = None  # Ordinary scoring was not requested for this cell.
            if not (is_reference and reference_on):
                diagnostics["reference_log_ratio_mean"][row] = None
        for key, values in diagnostics.items():
            batch.non_tensor_batch[f"feedback_{key}"] = np.array(values, dtype=object)

    def _log_rollout_data(self, batch, reward_extra_infos_dict, timing_raw, rollout_data_dir):
        extras = dict(reward_extra_infos_dict)
        extras.update({key: values.tolist() for key, values in batch.non_tensor_batch.items()
                       if key.startswith("feedback_")})
        return super()._log_rollout_data(batch, extras, timing_raw, rollout_data_dir)
