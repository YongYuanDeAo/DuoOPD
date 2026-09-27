"""Ordered training streams with a fixed number of questions per task in every update."""

import random


# Each task pool is shuffled with the data seed plus its own offset.
TASK_SEED_OFFSETS = {
    "biology_literqa": 101, "chemistry_literqa": 113, "physics_literqa": 127,
    "material_literqa": 163, "chemistry_understanding": 179, "physics_calculation": 191,
    "mbpp": 223,
}


def visit_order(size, per_step, steps, rng):
    """Pool indices visited by each update.

    The pool is read in one shuffled order; when it is exhausted, it is reshuffled and
    questions already chosen for the current update move to the back, so an update
    never repeats a question.
    """
    indices = list(range(size))
    rng.shuffle(indices)
    remaining, visits = indices.copy(), []
    for _ in range(steps):
        chosen = []
        for _ in range(per_step):
            if not remaining:
                remaining = indices.copy()
                rng.shuffle(remaining)
                remaining.sort(key=lambda index: index in chosen)
            chosen.append(remaining.pop(0))
        visits.append(chosen)
    return visits


def task_stream(pools, per_task, steps, seed):
    """Concatenate `per_task` questions of each task (sorted by name), then permute each update."""
    tasks = sorted(pools)
    visits = {task: visit_order(len(pools[task]), per_task, steps,
                                random.Random(seed + TASK_SEED_OFFSETS[task]))
              for task in tasks}
    update_rng = random.Random(seed)
    stream = []
    for step in range(steps):
        update = [pools[task][index] for task in tasks for index in visits[task][step]]
        order = list(range(len(update)))
        update_rng.shuffle(order)
        stream.extend(update[index] for index in order)
    return stream
