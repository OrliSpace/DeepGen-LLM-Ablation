from mmengine.config import read_base
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from src.optimisers.custom_adamw import CustomAdamW


with read_base():
    from ..models.deepgen_sft_llm_adapter import model
    from ..datasets.deepgen_512_fix_pixels.joint_sft_dual_stream_hf_stream import train_dataloader


model.num_queries = 128
model.use_activation_checkpointing = True
model.freeze_transformer = True  # Keep DiT frozen, train only LLM adapter (~10M params) for ultra-fast convergence
model.lora_modules = None        # No LoRA overhead needed initially; adapter learns residual injection directly


accumulative_counts = 4
dataloader_num_workers = 2
max_iters = 50000
optim_type = CustomAdamW
lr = 1e-4
betas = (0.9, 0.95)
weight_decay = 0.01
max_norm = 1.0
warmup_ratio = 0.02

save_steps = 1000
save_total_limit = 2

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
        eta_min=1e-6,
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
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)
