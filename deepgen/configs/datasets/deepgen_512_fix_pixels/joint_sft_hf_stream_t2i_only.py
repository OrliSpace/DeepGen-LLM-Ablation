from mmengine.config import read_base


with read_base():
    from .t2i_sft_hf_stream import train_dataloader
