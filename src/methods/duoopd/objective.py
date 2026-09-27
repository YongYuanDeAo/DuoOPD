"""DuoOPD token feedback: four outcome cells, online mean and the ablation switches."""

import torch
import torch.nn.functional as F

DEFAULT_SWITCHES = {"outcome_cells": True, "online_mean": True, "teacher_reference": True,
                    "raw_reference_ratio": False, "opd_cell": None, "response_mean": False,
                    "online_mean_scope": "task"}


def resolve_switches(algorithm):
    """Fill the `algorithm.duoopd` switches and reject meaningless combinations."""
    given = dict(algorithm.get("duoopd") or {})
    if set(given) - set(DEFAULT_SWITCHES):
        raise ValueError(f"Unknown duoopd switches: {sorted(set(given) - set(DEFAULT_SWITCHES))}")
    switches = {**DEFAULT_SWITCHES, **given}
    if not switches["outcome_cells"]:
        if set(given) - {"outcome_cells"}:
            raise ValueError("outcome_cells=false is plain sampled-token OPD; omit the other duoopd switches")
        return {**{key: False for key in DEFAULT_SWITCHES}, "opd_cell": None, "online_mean_scope": None}
    if not switches["online_mean"]:
        # The sharing scope only applies to the online mean.
        if "online_mean_scope" in given:
            raise ValueError("online_mean_scope requires online_mean=true")
        switches["online_mean_scope"] = None
    elif switches["online_mean_scope"] not in ("batch", "task"):
        raise ValueError("online_mean_scope must be batch or task")
    if switches["opd_cell"] not in (None, "shared_success", "teacher_only", "student_only", "shared_failure"):
        raise ValueError("opd_cell must name one of the four teacher-student outcome cells")
    if switches["raw_reference_ratio"] and not switches["teacher_reference"]:
        raise ValueError("raw_reference_ratio requires teacher_reference: without a reference, dT equals d0")
    if switches["response_mean"] and switches["online_mean"]:
        raise ValueError("response_mean replaces the batch mean; set online_mean=false")
    return switches


@torch.no_grad()
def duoopd_weights(d, response_mask, student_correct, teacher_correct, switches, task_ids=None):
    """Stopped per-token weights and the batch-wide success mean (or None).

    `d` is the teacher-minus-student log ratio at each sampled token, scored with the
    teacher reference on teacher-correct/student-wrong answers when the reference is on.
    `teacher_correct` is None when the teacher cache is not loaded.
    Task-scoped means use full-rollout-batch task IDs, before actor sharding.
    """
    mask = response_mask.bool()
    d = d.float()
    if not switches["outcome_cells"]:
        return d.masked_fill(~mask, 0), None
    student = student_correct.bool()
    weights = torch.where(student[:, None], F.softplus(d), -F.softplus(-d))
    constant = None
    if teacher_correct is not None:
        teacher = teacher_correct.bool()
        if switches["raw_reference_ratio"]:
            weights = torch.where((teacher & ~student)[:, None], d, weights)
        success_cell = ~teacher & student
        if (switches["online_mean"] or switches["response_mean"]) and switches["opd_cell"] != "student_only" and success_cell.any():
            # Average valid tokens within each answer, then give answers equal weight.
            answer_means = F.softplus(d).masked_fill(~mask, 0).sum(-1) / mask.sum(-1)
            if switches["online_mean"]:
                if switches["online_mean_scope"] == "batch":
                    constant = answer_means[success_cell].mean().item()
                    weights = torch.where(success_cell[:, None], constant, weights)
                else:
                    if task_ids is None:
                        raise ValueError("Task-scoped online means require rollout task IDs")
                    for task in torch.unique(task_ids[success_cell]):
                        selected = success_cell & (task_ids == task)
                        task_mean = answer_means[selected].mean()
                        weights = torch.where(selected[:, None], task_mean, weights)
            else:
                weights = torch.where(success_cell[:, None], answer_means[:, None], weights)
        if switches["opd_cell"] is not None:
            selected = {
                "shared_success": teacher & student,
                "teacher_only": teacher & ~student,
                "student_only": ~teacher & student,
                "shared_failure": ~teacher & ~student,
            }[switches["opd_cell"]]
            # Teacher-only OPD is scored without a reference, so d is d0 here too.
            weights = torch.where(selected[:, None], d, weights)
    return weights.masked_fill(~mask, 0), constant
