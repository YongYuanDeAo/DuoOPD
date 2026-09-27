"""Binary task verifiers shared by training and evaluation.

- Science multiple choice: the final top-level \\boxed{...} must hold the gold letter.
- MBPP: the code runs against all original assertions in an offline Bubblewrap sandbox.
- Instruction following: official IFEval checks for IFEval prompts, Open-Instruct's
  RLVR-IFeval checks for training prompts.
"""

from __future__ import annotations

from functools import lru_cache
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


CHOICE_TASKS = (
    "biology_literqa", "chemistry_literqa", "physics_literqa",
    "material_literqa", "chemistry_understanding", "physics_calculation",
)
VERIFICATION_FAILURE_REASONS = (
    "missing_box",
    "malformed_box",
    "invalid_task_answer",
    "incorrect_answer",
    "constraint_failed",
    "execution_timeout",
    "execution_error",
)
EXTERNAL = Path(__file__).resolve().parents[2] / "external"


def verify_answer(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict | None = None,
    **_: object,
) -> dict[str, float | str | None]:
    task = str(data_source).strip().lower()
    if task not in (*CHOICE_TASKS, "mbpp", "ifeval"):
        raise ValueError(f"unsupported task: {task!r}")
    if not isinstance(ground_truth, dict) or ground_truth.get("task") != task:
        raise ValueError("data source and ground-truth task do not match")
    if task == "mbpp":
        return verify_code(solution_str, ground_truth)
    if task == "ifeval":
        return verify_instructions(solution_str, ground_truth["target"])
    return verify_choice(task, solution_str, ground_truth["target"])


# Science multiple choice

_BOXED_COMMAND = re.compile(r"\\boxed(?![A-Za-z])")


def verify_choice(task, response, target):
    answer, box_failure = _extract_boxed_answer(response)
    normalized, target = _choice(answer), _choice(target)
    if target is None:
        raise ValueError(f"invalid {task} ground truth")
    if box_failure is not None:
        failure_reason = box_failure
    elif normalized is None:
        failure_reason = "invalid_task_answer"
    elif normalized != target:
        failure_reason = "incorrect_answer"
    else:
        failure_reason = None
    return {
        "score": float(failure_reason is None),
        "normalized_answer": normalized,
        "failure_reason": failure_reason,
    }


def _extract_boxed_answer(text: str) -> tuple[str | None, str | None]:
    """Read the final top-level box; malformed or nested answers are invalid."""

    value = str(text)
    commands = list(_BOXED_COMMAND.finditer(value))
    if not commands:
        return None, "missing_box"

    answer = None
    box_end = 0
    for command in commands:
        if command.start() < box_end:
            continue
        answer = None
        opening = command.end()
        while opening < len(value) and value[opening].isspace():
            opening += 1
        boxed = _braced_content(value, opening)
        if boxed is None:
            # An unclosed box contains any later boxed command.
            if opening < len(value) and value[opening] == "{":
                break
            continue
        content, box_end = boxed
        if _BOXED_COMMAND.search(content) is None:
            answer = content.strip()
    if not answer:
        return None, "malformed_box"
    return answer, None


def _braced_content(value: str, opening: int) -> tuple[str, int] | None:
    if opening >= len(value) or value[opening] != "{":
        return None
    depth = 0
    for position in range(opening, len(value)):
        if value[position] == "{":
            depth += 1
        elif value[position] == "}":
            depth -= 1
            if depth == 0:
                return value[opening + 1 : position], position + 1
    return None


def _choice(value: object) -> str | None:
    value = str(value).strip()
    value = re.sub(r"\\(?:text|mathrm|textrm|textbf)\{\s*([A-Da-d])\s*\}", r"\1", value)
    return value.upper() if re.fullmatch(r"[A-Da-d]", value) else None


# MBPP

# The child exposes only the Python runtime and system libraries. Candidate output
# goes to /dev/null; this descriptor carries a bounded verdict to the parent.
_RUNNER = r'''
import json, os, resource, sys
payload = json.load(sys.stdin)
result_fd = os.dup(1)
with open('/dev/null', 'w') as sink:
    os.dup2(sink.fileno(), 1)
    os.dup2(sink.fileno(), 2)
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
os.write(result_fd, b'ready\n')
namespace = {'__name__': '__main__'}
try:
    exec(compile(payload['code'], '<candidate>', 'exec'), namespace)
    exec(compile(payload['test_setup_code'], '<setup>', 'exec'), namespace)
    for test in payload['test_list']:
        exec(compile(test, '<test>', 'exec'), namespace)
except AssertionError:
    verdict = {'failure_reason': 'incorrect_answer'}
except BaseException as error:
    verdict = {'failure_reason': 'execution_error',
               'execution_detail': type(error).__name__ + ': ' + str(error)[:160]}
else:
    verdict = {'failure_reason': None}
os.write(result_fd, (json.dumps(verdict) + '\n').encode())
'''


def verify_code(response, ground_truth):
    tests = ground_truth["test_list"]
    if not tests:
        raise ValueError("MBPP ground truth has no execution tests")
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", response, flags=re.S | re.I)
    if "```" in response:
        code = blocks[0] if len(blocks) == 1 and response.count("```") == 2 else ""
    else:
        code = response.strip()
    if not code.strip():
        return {"score": 0.0, "normalized_answer": None, "failure_reason": "invalid_task_answer"}

    command = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv"]
    for directory in dict.fromkeys(("/usr", "/lib", "/lib64", sys.base_prefix, sys.prefix)):
        if Path(directory).exists():
            command += ["--ro-bind", directory, directory]
    command += ["--proc", "/proc", "--dev", "/dev", "--dir", "/tmp", "--chdir", "/tmp",
                "--remount-ro", "/", "--remount-ro", "/dev",
                "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "OMP_NUM_THREADS", "1",
                sys.executable, "-I", "-B", "-c", _RUNNER]
    payload = json.dumps({"code": code, "test_list": tests,
                          "test_setup_code": ground_truth.get("test_setup_code") or ""})
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        timed_out = False
        try:
            output, errors = process.communicate(payload, timeout=5)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            output, errors = process.communicate()
    if not output.startswith("ready\n"):
        raise RuntimeError(f"MBPP sandbox failed to start: {errors[:300]}")
    verdict_text = output.removeprefix("ready\n").strip()
    if timed_out or process.returncode in (-signal.SIGKILL, -signal.SIGXCPU, 137, 152):
        verdict = {"failure_reason": "execution_timeout"}
    elif process.returncode or not verdict_text:
        verdict = {"failure_reason": "execution_error"}
    else:
        verdict = json.loads(verdict_text)
    return {"score": float(verdict["failure_reason"] is None), "normalized_answer": None,
            "failure_reason": verdict["failure_reason"]}


# Instruction following

@lru_cache(maxsize=1)
def official_ifeval():
    sys.path.insert(0, str(EXTERNAL / "google-research"))
    from langdetect import DetectorFactory
    from instruction_following_eval import evaluation_lib

    DetectorFactory.seed = 0
    return evaluation_lib


@lru_cache(maxsize=1)
def if_functions():
    path = EXTERNAL / "open-instruct/open_instruct/if_functions.py"
    spec = importlib.util.spec_from_file_location("open_instruct_if_functions", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IF_FUNCTIONS_MAP


def if_check(target):
    """Resolve an RLVR-IFeval label to its checker and bound arguments."""
    parameters = json.loads(target)
    function = if_functions()[parameters.pop("func_name")]
    arguments = {key: value for key, value in parameters.items() if value is not None}
    inspect.signature(function).bind("", **arguments)
    return function, arguments


def verify_instructions(response, target):
    label = json.loads(target)
    if "instruction_id_list" in label:
        library = official_ifeval()
        item = library.InputExample(**label)
        responses = {item.prompt: response}
        strict = library.test_instruction_following_strict(item, responses)
        loose = library.test_instruction_following_loose(item, responses)
        correct = strict.follow_all_instructions
        return {"score": float(correct), "normalized_answer": None,
                "failure_reason": None if correct else "constraint_failed",
                "ifeval": {"prompt_strict": correct,
                           "prompt_loose": loose.follow_all_instructions,
                           "instruction_strict": strict.follow_instruction_list,
                           "instruction_loose": loose.follow_instruction_list}}
    function, arguments = if_check(target)
    if function.__name__ in ("validate_uppercase", "validate_lowercase"):
        # Upstream equality with upper()/lower() also accepts digits alone.
        correct = response.isupper() if function.__name__ == "validate_uppercase" else response.islower()
    else:
        correct = bool(response.strip()) and bool(function(response, **arguments))
    return {"score": float(correct), "normalized_answer": None,
            "failure_reason": None if correct else "constraint_failed"}
