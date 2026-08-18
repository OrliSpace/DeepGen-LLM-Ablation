from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler
from src.datasets.text2image.hf_streaming_datasets import HFStreamingT2IDataset, HFStreamingEditingDataset, HFStreamingJointDataset
from src.datasets.collate_functions import collate_func_gen_txt_dynamic, collate_func_img2img_txt_dynamic, CollateConcat


with read_base():
    from .processors import image_size, image_process


# Stream 1: Text-to-Image (T2I)
t2i_dataset = dict(
    type=HFStreamingT2IDataset,
    dataset_name='conceptual_captions',
    split='train',
    config_name=None,
    image_url_key='image_url',
    caption_key='caption',
    image_size=image_size,
    image_process=image_process,
    unit_image_size=32,
    virtual_size=200000,
    request_timeout=15,
)

# Stream 2: Image Editing (I2I)
edit_dataset = dict(
    type=HFStreamingEditingDataset,
    dataset_name='iitolstykh/NHR-Edit',
    split='train',
    source_image_key='source_image',
    target_image_key='edited_image',
    instruction_key='edit_instruction',
    image_size=image_size,
    image_process=image_process,
    unit_image_size=32,
    virtual_size=200000,
    request_timeout=15,
)

dataset = dict(
    type=HFStreamingJointDataset,
    t2i_dataset=t2i_dataset,
    edit_dataset=edit_dataset,
    t2i_ratio=0.5,
    virtual_size=400000,
)

collate_fn = dict(
    type=CollateConcat,
    keys=['text2image', 'image2image'],
    collate_fns=[
        dict(type=collate_func_gen_txt_dynamic),
        dict(type=collate_func_img2img_txt_dynamic)
    ]
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    pin_memory=True,
    persistent_workers=False,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=collate_fn,
)
