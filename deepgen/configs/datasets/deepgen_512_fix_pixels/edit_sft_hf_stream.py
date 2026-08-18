from ..hf_streaming_datasets import data_args as base_data_args

# This is a conceptual example. The actual field names in the NHR-Edit dataset
# and the required keys for the model's editing task need to be verified.
# The data adapter in `deepgen/src/datasets/text2image/hf_streaming_datasets.py`
# would likely need to be extended to handle this new schema.
#
# NHR-Edit on Hugging Face: https://huggingface.co/datasets/iitolstykh/NHR-Edit
# It contains triplets. We need to map its fields to the keys expected by the training pipeline,
# for example: 'source_pixel_values', 'pixel_values' (target), and 'text'.

data_args = base_data_args.copy()
data_args.update(
    dataset_name="iitolstykh/NHR-Edit",
    image_field="output_image",  # Hypothetical field for the target image
    source_image_field="input_image", # Hypothetical field for the source image
    caption_field="instruction", # Hypothetical field for the edit instruction
    train_split="train", # Check the dataset card for correct split names
    data_type="imageedit", # Set type to 'imageedit' for the collator
)

dataset = dict(
    type="HFStreamingDataset",
    data_args=data_args,
)