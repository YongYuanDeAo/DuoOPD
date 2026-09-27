"""Keep reference lookup identity beside the native verl prompt."""

from verl.utils.dataset.rl_dataset import RLHFDataset


class ReferenceDataset(RLHFDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        # The native trainer removes raw_prompt/example_id for rollout generation.
        row["extra_info"] = {**row["extra_info"], "example_id": row["example_id"],
                             "messages": row["raw_prompt"]}
        return row
