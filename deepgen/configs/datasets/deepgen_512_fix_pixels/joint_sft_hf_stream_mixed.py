from .t2i_sft_hf_stream import dataset as t2i_dataset
from .edit_sft_hf_stream import dataset as edit_dataset

# This config combines multiple streaming datasets.
# The training script and data collator must support handling a list of datasets
# and correctly mixing batches from different sources.

# Example of combining with sampling probabilities (weights).
# The dataloader would need to be configured to respect these weights.
datasets = [
    dict(
        dataset=t2i_dataset,
        weight=0.5,
    ),
    dict(
        dataset=edit_dataset,
        weight=0.5,
    )
]

# The final dataset object passed to the trainer would be a composite dataset.
# This might require a custom `WeightedConcatHFStreamingDataset` class that
# handles weighted sampling from multiple streaming sources.
dataset = dict(
    type="WeightedConcatHFStreamingDataset", # This is a hypothetical type
    datasets=datasets,
)