"""Score actual student tokens with a frozen native verl teacher worker, routing teacher references."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.model import compute_position_id_with_mask

from data import plain_nested
from evaluation.verifier import verify_answer


def reference_prompt(tokenizer, messages, reference, template_kwargs):
    """Append a reference to the teacher's last user message, before its answer."""
    contextual = deepcopy(messages)
    contextual[-1]["content"] += (
        "\n\nUse the following verified reference to solve the question.\n"
        "<reference>\n" + reference + "\n</reference>"
    )
    return tokenizer.apply_chat_template(
        contextual, tokenize=True, add_generation_prompt=True, **template_kwargs
    )


def prepare_teacher_batch(data, tokenizer, references, settings):
    """Grade original responses, route references, and preserve response token IDs."""
    batch = data.batch
    mask = batch["response_mask"].bool()
    response_width = mask.shape[-1]
    prompt_mask = batch["attention_mask"][:, :-response_width].bool()
    original_prompts = batch["input_ids"][:, :-response_width]
    contexts, outcomes, teacher_outcomes, branches, added_tokens, example_ids = [], [], [], [], [], []
    template_kwargs = dict(settings.apply_chat_template_kwargs)
    for row in range(len(data)):
        metadata = plain_nested(data.non_tensor_batch["extra_info"][row])
        example_id = str(metadata["example_id"])
        task = str(data.non_tensor_batch["data_source"][row])
        target = plain_nested(data.non_tensor_batch["reward_model"][row])["ground_truth"]
        teacher_sample = None
        teacher_success = -1.0
        if references is not None:
            group = references[example_id]
            if group["ground_truth"] != target or group["task"] != task:
                raise ValueError(f"Teacher reference does not match training question {example_id}")
            teacher_sample = group["samples"][0]
            teacher_success = float(teacher_sample["score"]["score"])
        response = tokenizer.decode(batch["responses"][row][mask[row]].tolist(), skip_special_tokens=True)
        success = float(verify_answer(task, response, target)["score"])
        if success not in (0, 1) or (references is not None and teacher_success not in (0, 1)):
            raise ValueError("Reference routing requires binary verifier outcomes")
        prompt = original_prompts[row][prompt_mask[row]].tolist()
        if success or references is None:
            context, branch = prompt, "ordinary"
        elif teacher_success and settings.use_teacher_answers:
            context = reference_prompt(tokenizer, metadata["messages"], teacher_sample["response"], template_kwargs)
            branch = "teacher_answer"
        else:
            context = prompt
            branch = "original_teacher_correct" if teacher_success else "original_teacher_wrong"
        if len(context) > settings.max_prompt_length:
            raise ValueError(f"Teacher context exceeds token budget for {example_id}; references are not truncated")
        contexts.append(context)
        outcomes.append(success)
        teacher_outcomes.append(teacher_success)
        branches.append(branch)
        added_tokens.append(len(context) - len(prompt))
        example_ids.append(example_id)

    width = max(map(len, contexts))
    ids = torch.full((len(data), width + response_width), tokenizer.pad_token_id,
                     dtype=batch["input_ids"].dtype, device=batch["input_ids"].device)
    attention = torch.zeros_like(ids)
    for row, context in enumerate(contexts):
        ids[row, width - len(context):width] = torch.tensor(context, device=ids.device)
        attention[row, width - len(context):width] = 1
    ids[:, width:] = batch["responses"]
    attention[:, width:] = mask
    teacher_data = DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": attention,
        "position_ids": compute_position_id_with_mask(attention),
        "responses": batch["responses"], "response_mask": batch["response_mask"],
    }, meta_info=dict(data.meta_info))
    true_rewards = torch.zeros_like(batch["responses"], dtype=torch.float32)
    lengths = mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("Cannot score an empty student response")
    true_rewards[torch.arange(len(data), device=ids.device), lengths - 1] = torch.tensor(
        outcomes, dtype=torch.float32, device=ids.device
    )
    diagnostics = {
        "reference_example_id": np.array(example_ids, dtype=object),
        "reference_branch": np.array(branches, dtype=object),
        "reference_teacher_correct": np.array(teacher_outcomes),
        "reference_student_correct": np.array(outcomes),
        "reference_added_tokens": np.array(added_tokens),
    }
    return teacher_data, true_rewards, diagnostics


class TeacherWorker(ActorRolloutRefWorker):
    def __init__(self, config):
        self.reference_settings = config.reference
        self.references = None
        if config.load_teacher_cache:
            self.references = {}
            with Path(config.reference.path).open() as handle:
                for line in handle:
                    group = json.loads(line)
                    if group["example_id"] in self.references or len(group["samples"]) != 1:
                        raise ValueError("Expected one fixed teacher response per training question")
                    self.references[group["example_id"]] = group
        # Use verl's causal-LM reference scorer, in the colocated reward role.
        # These are native worker fields, derived from the run's teacher settings.
        teacher_config = OmegaConf.create({
            "model": config.model,
            "actor": {"strategy": config.strategy, "fsdp_config": config.model.fsdp_config,
                      "ulysses_sequence_parallel_size": config.ulysses_sequence_parallel_size},
            "ref": {
                "fsdp_config": config.model.fsdp_config,
                "log_prob_micro_batch_size": config.micro_batch_size,
                "log_prob_micro_batch_size_per_gpu": config.micro_batch_size_per_gpu,
                "log_prob_use_dynamic_bsz": config.use_dynamic_bsz,
                "log_prob_max_token_len_per_gpu": config.forward_max_token_len_per_gpu,
                "ulysses_sequence_parallel_size": config.ulysses_sequence_parallel_size,
                "entropy_from_logits_with_chunking": False,
                "entropy_checkpointing": False,
                "profiler": config.profiler,
            },
            "rollout": {"temperature": config.teacher_temperature},
        })
        super().__init__(teacher_config, role="ref")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        self.ref_module_fsdp.eval().requires_grad_(False)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_rm_score(self, data):
        teacher_data, outcomes, diagnostics = prepare_teacher_batch(
            data, self.tokenizer, self.references, self.reference_settings
        )
        if teacher_data.batch["input_ids"].shape[-1] > self.ref_module_fsdp.config.max_position_embeddings:
            raise ValueError("Teacher scoring exceeds model context length")
        scored = super().compute_ref_log_prob(teacher_data)
        log_probs = scored.batch["ref_log_prob"].float()
        log_probs = log_probs.masked_fill(~data.batch["response_mask"].cpu().bool(), 0)
        tensors = {"teacher_log_probs": log_probs, "rm_scores": outcomes.cpu()}
        # Native rollout logs use the verifier score. The trainer builds the
        # log-ratio only after native old-policy scoring has completed.
        return DataProto.from_dict(
            tensors=tensors,
            non_tensors=diagnostics,
        )
