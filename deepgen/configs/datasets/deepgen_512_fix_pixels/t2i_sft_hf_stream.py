from mmengine.config import read_base
from mmengine.dataset import InfiniteSampler

from src.datasets.collate_functions import collate_func_gen_txt_dynamic
from src.datasets.text2image.hf_streaming_datasets import HFStreamingT2IDataset


with read_base():
    from .processors import image_size, image_process


dataset = dict(
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


train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    pin_memory=True,
    persistent_workers=False,
    dataset=dataset,
    sampler=dict(type=InfiniteSampler, shuffle=True),
    collate_fn=dict(type=collate_func_gen_txt_dynamic),
)
