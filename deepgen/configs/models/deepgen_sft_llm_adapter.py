import os
import torch
from src.models.sd3_kontext.deepgen_sft_llm_adapter import DeepGenSFTLLMAdapter
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from src.models.sd3_kontext.transformer_sd3_dynamic import SD3Transformer2DModel


sd3_5_model_name_or_path = "model_zoo/UniPic2-SD3.5M-Kontext-2B"
qwen2_5_vl_model_name_or_path = "model_zoo/Qwen2.5-VL-3B-Instruct"
qwen2_5_llm_model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"

# SFT baseline checkpoint
sft_ckpt_path = "checkpoints/model.pt"

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
    trust_remote_code=True,
    padding_side='right')

prompt_template = dict(
    IMG_START_TOKEN='<|vision_start|>',
    IMG_END_TOKEN='<|vision_end|>',
    IMG_CONTEXT_TOKEN='<|image_pad|>',
    IMG_START_TOKEN_FOR_GENERATION=False,
    SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
    INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                 '<|im_start|>assistant\n'),
    SUFFIX='<|im_end|>',
    SUFFIX_AS_EOS=True,
    SEP='\n',
    STOP_WORDS=['<|im_end|>', '<|endoftext|>'],
    GENERATION='Generate an image: {input}',
    CFG='Generate an image.'
)

model = dict(
    type=DeepGenSFTLLMAdapter,
    num_queries=128,
    llm_name_or_path=qwen2_5_llm_model_name_or_path,
    freeze_llm=True,
    connector=dict(
        hidden_size=2048,
        intermediate_size=11946,
        num_hidden_layers=6,
        _attn_implementation='flash_attention_2',
        num_attention_heads=32,
    ),
    lmm=dict(
        type=Qwen2_5_VLForConditionalGeneration.from_pretrained,
        pretrained_model_name_or_path=qwen2_5_vl_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ),
    tokenizer=tokenizer,
    prompt_template=prompt_template,
    freeze_lmm=True,
    freeze_mq=False,
    transformer=dict(
        type=SD3Transformer2DModel.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ),
    test_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="scheduler",
    ),
    train_scheduler=dict(
        type=FlowMatchEulerDiscreteScheduler.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="scheduler",
    ),
    vae=dict(
        type=AutoencoderKL.from_pretrained,
        pretrained_model_name_or_path=sd3_5_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    ),
    pretrained_pth=sft_ckpt_path,
    use_activation_checkpointing=True,
    freeze_transformer=False,
    scb_layers_vlm=[4, 10, 16, 22, 28, 35],
)
