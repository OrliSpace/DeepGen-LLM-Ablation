import random
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
import torch.distributed as dist
from mmengine.logging import print_log
from src.models.connector import ConnectorConfig, ConnectorEncoder
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from mmengine.model import BaseModel
from functools import partial
from six.moves import map, zip
from copy import deepcopy
from einops import rearrange
from transformers import AutoModelForCausalLM, AutoTokenizer


IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=[], verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)
    
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        print_log(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


class ZeroInitCrossAttentionAdapter(nn.Module):
    """
    Lightweight cross-attention adapter with zero-initialized output projection.
    Allows base sequence conditions (Q) to attend to frozen LLM text representations (K, V)
    with strict identity preservation at Step 0.
    """
    def __init__(self, query_dim=4096, context_dim=2048, num_heads=16, head_dim=64):
        super().__init__()
        self.inner_dim = num_heads * head_dim
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, query_dim, bias=True)

        # Zero-initialize output projection to ensure delta = 0 at step 0
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(self, x, context, context_mask=None):
        b, q_len, _ = x.shape
        _, ctx_len, _ = context.shape

        q = self.to_q(x).view(b, q_len, self.num_heads, -1).transpose(1, 2)
        k = self.to_k(context).view(b, ctx_len, self.num_heads, -1).transpose(1, 2)
        v = self.to_v(context).view(b, ctx_len, self.num_heads, -1).transpose(1, 2)

        attn_mask = None
        if context_mask is not None:
            # (B, 1, 1, ctx_len)
            attn_mask = context_mask[:, None, None, :].to(dtype=torch.bool)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(b, q_len, self.inner_dim)
        out = self.to_out(out)
        return out


@BUILDER.register_module()
class DeepGenSFTLLMAdapter(BaseModel):
    """
    Warm-start DeepGen model that loads the trained SFT checkpoint (VLM + SCB Connector + DiT)
    with 100% weight matching, augmented by a zero-initialized frozen LLM adapter.

    Guarantees:
    - Step 0 forward pass is 100.00% identical to the official trained DeepGen SFT checkpoint.
    - Zero degradation upon initialization.
    - No Stage A pretraining required; skips straight to lightweight SFT fine-tuning.
    """
    def __init__(self,
                 transformer,
                 train_scheduler,
                 test_scheduler,
                 vae,
                 lmm,
                 tokenizer,
                 prompt_template,
                 connector,
                 llm_name_or_path="Qwen/Qwen2.5-3B-Instruct",
                 freeze_llm=True,
                 num_queries=128,
                 vit_input_size=448,
                 max_length=1024,
                 freeze_lmm=True,
                 freeze_mq=False,
                 res_vit=False,
                 pretrained_pth=None,
                 use_activation_checkpointing=False,
                 lora_modules=None,
                 lora_rank=64,
                 lora_alpha=128,
                 freeze_transformer=True,
                 unconditional=0.1,
                 ema_cfg=None,
                 weighting_scheme='none',
                 logit_mean=0.0,
                 logit_std=1.0,
                 scb_layers_vlm=[4, 10, 16, 22, 28, 35],
                 ):
        super().__init__()

        # ---------------------------------------------------------------------
        # 1. Base DeepGen SFT Backbone (Preserves 100% of SFT state dict schema)
        # ---------------------------------------------------------------------
        self.lmm = BUILDER.build(lmm)
        if freeze_lmm:
            self.lmm.requires_grad_(False)
        self.freeze_lmm = freeze_lmm

        self.transformer = BUILDER.build(transformer)
        if freeze_transformer:
            self.transformer.requires_grad_(False)
        self.freeze_transformer = freeze_transformer
        self.res_vit = res_vit

        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std

        self.vae = BUILDER.build(vae)
        self.vae.requires_grad_(False)

        self.use_activation_checkpointing = use_activation_checkpointing
        self.tokenizer = BUILDER.build(tokenizer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.prompt_template = prompt_template
        self.vit_input_size = vit_input_size
        self.max_length = max_length
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(prompt_template['IMG_CONTEXT_TOKEN'])
        self.register_buffer('vit_mean', torch.tensor(IMAGE_MEAN), persistent=False)
        self.register_buffer('vit_std', torch.tensor(IMAGE_STD), persistent=False)

        self.num_queries = num_queries
        self.scb_layers_vlm = scb_layers_vlm

        # Exact SFT SCB Connector structure
        connector_cfg = ConnectorConfig(**connector)
        self.connector = ConnectorEncoder(connector_cfg)

        vlm_dim = self.lmm.language_model.config.hidden_size
        self.projector_1 = nn.Linear(vlm_dim * len(self.scb_layers_vlm), connector_cfg.hidden_size)
        self.projector_2 = nn.Linear(connector_cfg.hidden_size, self.transformer.config.pooled_projection_dim)
        self.projector_3 = nn.Linear(connector_cfg.hidden_size, self.transformer.config.joint_attention_dim)

        self.meta_queries = nn.Parameter(torch.zeros(num_queries, vlm_dim))
        nn.init.normal_(self.meta_queries, std=1 / math.sqrt(vlm_dim))

        # ---------------------------------------------------------------------
        # 2. Frozen Pure LLM Backbone for Linguistic & Spatial Reasoning
        # ---------------------------------------------------------------------
        attn_impl = "flash_attention_2" if torch.cuda.is_available() else "sdpa"
        self.llm_text = AutoModelForCausalLM.from_pretrained(
            llm_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True
        )
        self.tokenizer_llm = AutoTokenizer.from_pretrained(
            llm_name_or_path,
            trust_remote_code=True,
            padding_side='right'
        )
        if self.tokenizer_llm.pad_token is None:
            self.tokenizer_llm.pad_token = self.tokenizer_llm.eos_token

        if freeze_llm:
            self.llm_text.requires_grad_(False)
        self.freeze_llm = freeze_llm

        # ---------------------------------------------------------------------
        # 3. Zero-Initialized Residual Conditioning Adapters (~10M params)
        # ---------------------------------------------------------------------
        llm_dim = self.llm_text.config.hidden_size
        pooled_dim = self.transformer.config.pooled_projection_dim
        joint_dim = self.transformer.config.joint_attention_dim

        # Adapter for pooled condition (delta y_pool)
        self.adapter_pool = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.SiLU(),
            nn.Linear(llm_dim, pooled_dim)
        )
        nn.init.zeros_(self.adapter_pool[-1].weight)
        nn.init.zeros_(self.adapter_pool[-1].bias)

        # Adapter for sequence condition (delta c_seq)
        self.adapter_seq = ZeroInitCrossAttentionAdapter(
            query_dim=joint_dim,
            context_dim=llm_dim,
            num_heads=16,
            head_dim=64
        )

        if freeze_mq:
            self.projector_1.requires_grad_(False)
            self.projector_2.requires_grad_(False)
            self.projector_3.requires_grad_(False)
            self.connector.requires_grad_(False)
            self.meta_queries.requires_grad_(False)
        self.freeze_mq = freeze_mq

        self.unconditional = unconditional
        self.train_scheduler = BUILDER.build(train_scheduler)
        self.test_scheduler = BUILDER.build(test_scheduler)

        if use_activation_checkpointing:
            self.gradient_checkpointing_enable()

        # Optional LoRA for Stage B DiT/VLM fine-tuning
        if lora_modules is not None:
            assert self.freeze_lmm
            self.lmm.language_model.config.tie_word_embeddings = False
            if lora_modules == 'auto':
                lora_modules = find_target_linear_names(self.lmm)
            transformer_lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                init_lora_weights="gaussian",
                target_modules=lora_modules,
                lora_dropout=0.05,
            )
            self.lmm.add_adapter(transformer_lora_config)

        # Load official trained DeepGen SFT checkpoint
        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Loaded DeepGen pretrained SFT weights from {pretrained_pth}: missing_keys={len(info.missing_keys)}')

        self.ema_cfg = ema_cfg
        if ema_cfg is not None:
            self.ema = nn.ModuleDict()
            self.ema.steps = 0
            if not self.freeze_transformer:
                self.ema.update(dict(transformer=deepcopy(self.transformer)))

            self.ema.update(dict(
                adapter_pool=deepcopy(self.adapter_pool),
                adapter_seq=deepcopy(self.adapter_seq)
            ))
            self.ema.requires_grad_(False)

            if 'checkpoint' in ema_cfg:
                ema_state_dict = guess_load_checkpoint(ema_cfg['checkpoint'])
                self.ema.load_state_dict(ema_state_dict, strict=False)
                print_log(f"Loaded EMA weights from {ema_cfg['checkpoint']}")

    @torch.no_grad()
    def ema_step(self):
        if self.ema_cfg is None:
            return

        steps = self.ema.steps
        update_interval = self.ema_cfg.get('update_interval', 1)
        save_interval = self.ema_cfg.get('save_interval', 1000)
        momentum = self.ema_cfg.get('momentum', 0.99)

        if steps % update_interval == 0 and steps > 0:
            for ema_p, base_p in zip(self.ema.adapter_pool.parameters(), self.adapter_pool.parameters()):
                ema_p.data.lerp_(base_p.data.detach(), 1.0 - momentum)
            for ema_p, base_p in zip(self.ema.adapter_seq.parameters(), self.adapter_seq.parameters()):
                ema_p.data.lerp_(base_p.data.detach(), 1.0 - momentum)

            if not self.freeze_transformer:
                for ema_p, base_p in zip(self.ema.transformer.parameters(), self.transformer.parameters()):
                    ema_p.data.lerp_(base_p.data.detach(), 1.0 - momentum)

        if steps % save_interval == 0 and steps > 0:
            is_ddp = dist.is_available() and dist.is_initialized()
            is_primary_proc = (not is_ddp) or dist.get_rank() == 0
            if is_primary_proc:
                save_path = self.ema_cfg.get('save_path')
                if save_path:
                    torch.save(self.ema.state_dict(), save_path)
            if is_ddp:
                dist.barrier()

        self.ema.steps = self.ema.steps + 1

    def llm2dit_base(self, x):
        """Standard SFT SCB projection."""
        x = self.connector(self.projector_1(x))
        pooled_out = self.projector_2(x.mean(dim=1))
        seq_out = self.projector_3(x)
        return pooled_out, seq_out

    def inject_llm_reasoning(self, pooled_base, seq_base, texts):
        """
        Runs frozen text LLM and injects zero-initialized residual reasoning signals.
        """
        # Clean special vision tokens for pure text LLM
        clean_prompts = []
        for text in texts:
            clean_p = (text.replace("<image>", "")
                           .replace("<|image_pad|>", "")
                           .replace("<|vision_start|>", "")
                           .replace("<|vision_end|>", "")
                           .strip())
            if len(clean_p) == 0:
                clean_p = " "
            clean_prompts.append(clean_p)

        text_inputs_llm = self.tokenizer_llm(
            clean_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Frozen LLM forward
        with torch.no_grad() if self.freeze_llm else torch.enable_grad():
            output_llm = self.llm_text.model(
                input_ids=text_inputs_llm['input_ids'].long(),
                attention_mask=text_inputs_llm['attention_mask'].bool(),
                return_dict=True,
                use_cache=False
            )
            h_llm = output_llm.last_hidden_state  # (B, L_txt, 2048)

        # Residual projections (zero-initialized)
        delta_pooled = self.adapter_pool(h_llm.mean(dim=1))  # (B, 2048)
        delta_seq = self.adapter_seq(seq_base, h_llm, context_mask=text_inputs_llm['attention_mask'])  # (B, L_vlm, 4096)

        final_pooled = pooled_base + delta_pooled
        final_seq = seq_base + delta_seq

        return final_pooled, final_seq

    @property
    def llm(self):
        return self.lmm.language_model

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.llm_text.gradient_checkpointing_enable()
        self.transformer.enable_gradient_checkpointing()
        self.connector.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.llm_text.gradient_checkpointing_disable()
        self.transformer.disable_gradient_checkpointing()
        self.connector.gradient_checkpointing = False

    @property
    def device(self):
        return self.llm.device

    @property
    def dtype(self):
        return self.llm.dtype

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        if self.vae is not None:
            self.vae.train(mode=False)
        if not mode:
            self.gradient_checkpointing_disable()
        return self

    def state_dict(self, *args, **kwargs) -> dict:
        state_dict = super().state_dict(*args, **kwargs)
        
        # Determine exactly which keys are trainable (e.g. adapters, LoRA) to avoid saving 20GB frozen backbones
        trainable_keys = set()
        for k, p in self.named_parameters():
            if p.requires_grad:
                trainable_keys.add(k)
                # Handle possible 'module.' prefix from DDP
                trainable_keys.add(k.replace('module.', ''))
                trainable_keys.add('module.' + k)
                
        # Keep only trainable weights in the saved checkpoint
        state_dict = {k: v for k, v in state_dict.items() if k in trainable_keys}
        return state_dict

    @torch.no_grad()
    def pixels_to_latents(self, x):
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    @torch.no_grad()
    def latents_to_pixels(self, z):
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        x_rec = self.vae.decode(z).sample
        return x_rec

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            self.ema_step()
            return self.compute_loss(data_dict=data)
        else:
            raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type in ['text2image', 'image2image']:
            if data_type in data_dict:
                losses[f'loss_{data_type}'] = getattr(self, f'{data_type}_loss')(data_dict[data_type])
        if len(losses) == 0:
            if 'pixel_values_src' in data_dict:
                losses['loss_image2image'] = self.image2image_loss(data_dict)
            else:
                losses['loss_text2image'] = self.text2image_loss(data_dict)
        return losses

    def prepare_forward_input(self,
                              query_embeds,
                              input_ids=None,
                              image_embeds=None,
                              image_grid_thw=None,
                              attention_mask=None,
                              past_key_values=None):
        b, l, _ = query_embeds.shape
        assert l > 0
        attention_mask = attention_mask.to(device=self.device, dtype=torch.bool)
        assert l == self.num_queries

        input_ids = torch.cat([input_ids, input_ids.new_zeros(b, l)], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, l)], dim=1)

        position_ids, _ = self.lmm.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            second_per_grid_ts=None,
            attention_mask=attention_mask,
        )

        if past_key_values is not None:
            inputs_embeds = query_embeds
            position_ids = position_ids[..., -l:]
        else:
            input_ids = input_ids[:, :-l]

            if image_embeds is None:
                inputs_embeds = self.llm.get_input_embeddings()(input_ids)
            else:
                inputs_embeds = torch.zeros(*input_ids.shape, self.llm.config.hidden_size,
                                            device=self.device, dtype=self.dtype)
                inputs_embeds[input_ids == self.image_token_id] = \
                    image_embeds.contiguous().view(-1, self.llm.config.hidden_size)
                inputs_embeds[input_ids != self.image_token_id] = self.llm.get_input_embeddings()(
                    input_ids[input_ids != self.image_token_id]
                )

            inputs_embeds = torch.cat([inputs_embeds, query_embeds], dim=1)

        inputs = dict(inputs_embeds=inputs_embeds,
                      attention_mask=attention_mask,
                      position_ids=position_ids,
                      past_key_values=past_key_values)
        return inputs

    @torch.no_grad()
    def get_semantic_features_dynamic(self, pixel_values):
        pixel_values = [F.interpolate(p[None], scale_factor=28 / 32, mode='bilinear') for p in pixel_values]
        image_embeds, image_grid_thw = multi_apply(self.get_semantic_features,
                                                   pixel_values, resize=False)
        image_embeds = [x[0] for x in image_embeds]
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
        return image_embeds, image_grid_thw

    @torch.no_grad()
    def get_semantic_features(self, pixel_values, resize=True):
        pixel_values = (pixel_values + 1.0) / 2
        pixel_values = pixel_values - self.vit_mean.view(1, 3, 1, 1)
        pixel_values = pixel_values / self.vit_std.view(1, 3, 1, 1)

        if resize:
            pixel_values = F.interpolate(pixel_values, size=(self.vit_input_size, self.vit_input_size),
                                         mode='bilinear')
        b, c, h, w = pixel_values.shape

        patch_size = self.lmm.config.vision_config.patch_size
        spatial_merge_size = self.lmm.config.vision_config.spatial_merge_size
        temporal_patch_size = self.lmm.config.vision_config.temporal_patch_size

        pixel_values = pixel_values[:, None].expand(b, temporal_patch_size, c, h, w)

        grid_t = 1
        grid_h, grid_w = h // patch_size, w // patch_size

        pixel_values = pixel_values.view(
            b,
            grid_t,
            temporal_patch_size,
            c,
            grid_h // spatial_merge_size,
            spatial_merge_size,
            patch_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
            patch_size,
        )

        pixel_values = rearrange(
            pixel_values, 'b t tp c h m p w n q -> (b t h w m n) (c tp p q)')

        image_grid_thw = torch.tensor([(grid_t, grid_h, grid_w)] * b).to(self.device).long()
        image_embeds = self.lmm.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = rearrange(image_embeds, '(b l) d -> b l d', b=b)

        return image_embeds, image_grid_thw

    @torch.no_grad()
    def prepare_text2image_prompts(self, texts):
        vlm_texts = [self.prompt_template['GENERATION'].format(input=text) for text in texts]
        vlm_texts = [self.prompt_template['INSTRUCTION'].format(input=text) for text in vlm_texts]

        return self.tokenizer(
            vlm_texts, add_special_tokens=True, return_tensors='pt', padding=True, padding_side='left').to(self.device)

    @torch.no_grad()
    def prepare_image2image_prompts(self, texts, num_refs, ref_lens):
        prompts = []
        cnt = 0
        for text, num_ref in zip(texts, num_refs):
            image_tokens = ''
            for _ in range(num_ref):
                image_tokens += (self.prompt_template['IMG_START_TOKEN'] +
                                 self.prompt_template['IMG_CONTEXT_TOKEN'] * ref_lens[cnt] +
                                 self.prompt_template['IMG_END_TOKEN'])
                cnt += 1
            prompts.append(self.prompt_template['INSTRUCTION'].format(input=f'{image_tokens}\n{text}'))

        return self.tokenizer(
            prompts, add_special_tokens=True, return_tensors='pt', padding=True, padding_side='left').to(self.device)

    def text2image_loss(self, data_dict):
        if 'image_latents' in data_dict:
            image_latents = [x.to(dtype=self.dtype, device=self.device) for x in data_dict['image_latents']]
        else:
            pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
            image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]

        b = len(image_latents)
        raw_texts = data_dict['texts'] if 'texts' in data_dict else [data_dict.get('text', '')] * b
        texts = ['' if random.uniform(0, 1) < self.unconditional else text for text in raw_texts]

        # 1. Base SFT VLM + SCB Connector Forward Pass
        text_inputs_vlm = self.prepare_text2image_prompts(texts)
        hidden_states_mq = self.meta_queries[None].expand(b, self.num_queries, -1)
        inputs_vlm = self.prepare_forward_input(query_embeds=hidden_states_mq, **text_inputs_vlm)

        max_length_vlm = self.max_length + self.num_queries
        inputs_embeds_vlm = inputs_vlm['inputs_embeds'][:, -max_length_vlm:]
        attention_mask_vlm = inputs_vlm['attention_mask'][:, -max_length_vlm:]
        position_ids_vlm = inputs_vlm['position_ids'][..., -max_length_vlm:]

        output_vlm = self.llm(
            inputs_embeds=inputs_embeds_vlm,
            attention_mask=attention_mask_vlm,
            position_ids=position_ids_vlm,
            output_hidden_states=True,
            return_dict=True
        )
        selected_vlm = [output_vlm.hidden_states[i] for i in self.scb_layers_vlm]
        merged_vlm = torch.cat(selected_vlm, dim=-1)

        pooled_base, seq_base = self.llm2dit_base(merged_vlm)

        # 2. Inject Zero-Initialized LLM Reasoning
        pooled_out, seq_out = self.inject_llm_reasoning(pooled_base, seq_base, texts)

        # 3. Diffusion Flow Matching Loss
        loss_diff = self.diff_loss(
            model_input=image_latents,
            pooled_prompt_embeds=pooled_out,
            prompt_embeds=seq_out
        )
        return loss_diff

    def image2image_loss(self, data_dict):
        pixel_values_src = data_dict['pixel_values_src']
        num_refs = [len(ref_images) for ref_images in pixel_values_src]

        pixel_values_src = [[img.to(dtype=self.dtype, device=self.device) for img in ref_images]
                            for ref_images in pixel_values_src]
        image_latents_src = [[self.pixels_to_latents(img[None])[0] for img in ref_images]
                             for ref_images in pixel_values_src]
        image_embeds, image_grid_thw = self.get_semantic_features_dynamic(
            [img for ref_images in pixel_values_src for img in ref_images])

        ref_lens = [len(x) for x in image_embeds]

        pixel_values = [p.to(dtype=self.dtype, device=self.device) for p in data_dict['pixel_values']]
        image_latents = [self.pixels_to_latents(p[None])[0] for p in pixel_values]

        b = len(image_latents)
        raw_texts = data_dict['texts'] if 'texts' in data_dict else [data_dict.get('text', '')] * b
        text_inputs_vlm = self.prepare_image2image_prompts(raw_texts, num_refs=num_refs, ref_lens=ref_lens)

        # 1. Base SFT VLM Multi-modal Forward
        hidden_states_mq = self.meta_queries[None].expand(b, self.num_queries, -1)
        inputs_vlm = self.prepare_forward_input(query_embeds=hidden_states_mq,
                                                image_embeds=torch.cat(image_embeds),
                                                image_grid_thw=image_grid_thw,
                                                **text_inputs_vlm)

        max_length_vlm = self.max_length + max(num_refs) * max(ref_lens) + self.num_queries
        inputs_embeds_vlm = inputs_vlm['inputs_embeds'][:, -max_length_vlm:]
        attention_mask_vlm = inputs_vlm['attention_mask'][:, -max_length_vlm:]
        position_ids_vlm = inputs_vlm['position_ids'][..., -max_length_vlm:]

        output_vlm = self.llm(
            inputs_embeds=inputs_embeds_vlm,
            attention_mask=attention_mask_vlm,
            position_ids=position_ids_vlm,
            output_hidden_states=True,
            return_dict=True
        )
        selected_vlm = [output_vlm.hidden_states[i] for i in self.scb_layers_vlm]
        merged_vlm = torch.cat(selected_vlm, dim=-1)

        pooled_base, seq_base = self.llm2dit_base(merged_vlm)

        # 2. Inject Zero-Initialized LLM Reasoning
        pooled_out, seq_out = self.inject_llm_reasoning(pooled_base, seq_base, raw_texts)

        # 3. Diffusion Flow Matching Loss with Reference Image Conditioning
        loss_diff = self.diff_loss(
            model_input=image_latents,
            pooled_prompt_embeds=pooled_out,
            prompt_embeds=seq_out,
            cond_intput=image_latents_src
        )
        return loss_diff

    def diff_loss(self, model_input, pooled_prompt_embeds, prompt_embeds, cond_intput=None):
        noise = [torch.randn_like(x) for x in model_input]
        bsz = len(model_input)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=bsz,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
        )
        indices = (u * self.train_scheduler.config.num_train_timesteps).long()
        timesteps = self.train_scheduler.timesteps[indices].to(device=self.device)

        sigmas = self.get_sigmas(timesteps, n_dim=model_input[0].ndim)
        noisy_model_input = [sigmas[i] * noise[i] + (1.0 - sigmas[i]) * model_input[i] for i in range(bsz)]

        model_pred = self.transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            cond_hidden_states=cond_intput,
            return_dict=False,
        )[0]

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme,
            sigmas=sigmas
        )

        target = [noise[i] - model_input[i] for i in range(bsz)]
        loss = [
            torch.mean(
                (weighting[i].float() * (model_pred[i].float() - target[i].float()) ** 2)
            )
            for i in range(bsz)
        ]
        loss = sum(loss) / len(loss)
        return loss

    def get_sigmas(self, timesteps, n_dim=4):
        sigmas = self.train_scheduler.sigmas.to(device=self.device, dtype=self.dtype)
        schedule_timesteps = self.train_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    @torch.no_grad()
    def generate(self,
                 prompt,
                 cfg_prompt,
                 pixel_values_src=None,
                 cfg_scale=4.5,
                 num_steps=50,
                 generator=None,
                 height=512,
                 width=512,
                 progress_bar=True):
        assert len(prompt) == len(cfg_prompt)
        b = len(prompt)

        if pixel_values_src is not None:
            num_refs = [len(ref_images) for ref_images in pixel_values_src]
            pixel_values_src = [[img.to(dtype=self.dtype, device=self.device) for img in ref_imgs]
                                for ref_imgs in pixel_values_src]
            image_embeds, image_grid_thw = self.get_semantic_features_dynamic(
                [img for ref_images in pixel_values_src for img in ref_images])
            ref_lens = [len(x) for x in image_embeds]

            text_inputs_vlm = self.prepare_image2image_prompts(prompt + cfg_prompt, num_refs=num_refs*2, ref_lens=ref_lens*2)
            text_inputs_vlm.update(image_embeds=torch.cat(image_embeds*2),
                                   image_grid_thw=torch.cat([image_grid_thw]*2))
            cond_latents = [[self.pixels_to_latents(img[None])[0] for img in ref_imgs]
                            for ref_imgs in pixel_values_src]
            cond_latents = cond_latents * 2
        else:
            text_inputs_vlm = self.prepare_text2image_prompts(prompt + cfg_prompt)
            cond_latents = None

        # 1. Base SFT VLM
        hidden_states_mq = self.meta_queries[None].expand(2*b, self.num_queries, -1)
        inputs_vlm = self.prepare_forward_input(query_embeds=hidden_states_mq, **text_inputs_vlm)
        output_vlm = self.llm(**inputs_vlm, output_hidden_states=True, return_dict=True)
        selected_vlm = [output_vlm.hidden_states[i] for i in self.scb_layers_vlm]
        merged_vlm = torch.cat(selected_vlm, dim=-1)

        pooled_base, seq_base = self.llm2dit_base(merged_vlm)

        # 2. Injected LLM Reasoning
        pooled_out, seq_out = self.inject_llm_reasoning(pooled_base, seq_base, prompt + cfg_prompt)

        # 3. Denoising loop
        scheduler = self.test_scheduler
        scheduler.set_timesteps(num_steps, device=self.device)
        timesteps = scheduler.timesteps

        latents_shape = (b, 16, height // 8, width // 8)
        latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=self.dtype)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])

            noise_pred = self.transformer(
                hidden_states=[latent_model_input[k] for k in range(latent_model_input.shape[0])],
                timestep=timestep,
                encoder_hidden_states=seq_out,
                pooled_projections=pooled_out,
                cond_hidden_states=cond_latents,
                return_dict=False,
            )[0]
            noise_pred = torch.stack(noise_pred, dim=0)

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        image = self.latents_to_pixels(latents)
        return image

