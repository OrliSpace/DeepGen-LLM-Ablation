from mmengine.config import read_base
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from src.optimisers.custom_adamw import CustomAdamW


with read_base():
    from ..models.deepgen_scb import model
    from ..datasets.deepgen_512_fix_pixels.joint_sft_hf_stream_t2i_only import train_dataloader


model.num_queries = 128
model.use_activation_checkpointing = False
model.freeze_transformer = False
model.lora_modules = 'auto'
model.lora_rank = 64
model.lora_alpha = 128
model.pretrained_pth = 'your_path_to_pretrained_pth'


accumulative_counts = 3
dataloader_num_workers = 4
max_iters = 400000
optim_type = CustomAdamW
lr = 5e-5
betas = (0.9, 0.95)
weight_decay = 0.05
max_norm = 1.0
warmup_ratio = 0.01


save_steps = 40000
save_total_limit = 5


optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16',
)


param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=warmup_ratio * max_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=False,
        begin=warmup_ratio * max_iters,
        end=max_iters)
]


train_cfg = dict(type=TrainLoop, max_iters=max_iters)


default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)


env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
