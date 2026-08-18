import io
import random
import numpy as np
import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from einops import rearrange

from src.datasets.utils import crop2square, resize_image_dynamic, resize_image_fix_pixels


class HFStreamingT2IDataset(Dataset):
    """Map-style adapter over Hugging Face streaming T2I datasets.

    Exposes an indexable interface for compatibility with InfiniteSampler/MultiSourceSampler
    while streaming samples on-the-fly without local disk caching.
    """

    def __init__(self,
                 dataset_name,
                 split='train',
                 config_name=None,
                 image_url_key='image_url',
                 caption_key='caption',
                 image_process='fix_pixels',
                 image_size=512,
                 unit_image_size=32,
                 virtual_size=200000,
                 request_timeout=15,
                 debug=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.image_url_key = image_url_key
        self.caption_key = caption_key
        self.image_process = image_process
        self.image_size = image_size
        self.unit_image_size = unit_image_size
        self.virtual_size = int(virtual_size)
        self.request_timeout = request_timeout
        self.debug = debug

        self._cache = []
        self._build_stream()

    def _build_stream(self):
        if self.config_name is None:
            self._stream = load_dataset(self.dataset_name, split=self.split, streaming=True)
        else:
            self._stream = load_dataset(self.dataset_name, self.config_name, split=self.split, streaming=True)
        self._iter = iter(self._stream)

    def _get_iter(self):
        if getattr(self, '_iter', None) is None:
            self._build_stream()
        return self._iter

    def __len__(self):
        return self.virtual_size

    def _process_image(self, image):
        if not isinstance(image, Image.Image):
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image)).convert('RGB')
            else:
                raise TypeError(f"Expected PIL Image or bytes, got {type(image)}")
        else:
            image = image.convert('RGB')

        if self.image_process == 'crop2square':
            image = crop2square(image)
            image = image.resize(size=(self.image_size, self.image_size))
        elif self.image_process == 'dynamic':
            image = resize_image_dynamic(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'fix_pixels':
            image = resize_image_fix_pixels(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'resize2square':
            image = image.resize(size=(self.image_size, self.image_size))
        else:
            raise NotImplementedError(f'Unsupported image_process: {self.image_process}')

        assert image.width % self.unit_image_size == 0
        assert image.height % self.unit_image_size == 0

        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def _next_metadata(self):
        it = self._get_iter()
        for _ in range(50):
            try:
                sample = next(it)
            except StopIteration:
                self._build_stream()
                it = self._get_iter()
                sample = next(it)
            except Exception as e:
                import time
                if self.debug:
                    print(f"[_next_metadata] Error during next(it): {e}. Rebuilding stream...", flush=True)
                time.sleep(2.0)
                self._build_stream()
                it = self._get_iter()
                sample = next(it)

            text = sample.get(self.caption_key)
            image_raw = sample.get(self.image_url_key)

            if text is not None and image_raw is not None:
                if isinstance(text, str) and text.strip():
                    return dict(text=text.strip(), image_raw=image_raw)

        raise RuntimeError(
            f'Failed to get valid sample from {self.dataset_name}/{self.config_name}. '
            f'Expected keys: text={self.caption_key}, image={self.image_url_key}'
        )

    def __getitem__(self, idx):
        import time
        last_exception = None
        for retry in range(10):
            try:
                sample = self._next_metadata()
                image_raw = sample['image_raw']
                if isinstance(image_raw, str) and (image_raw.startswith('http://') or image_raw.startswith('https://')):
                    response = requests.get(image_raw, timeout=self.request_timeout)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content)).convert('RGB')
                elif isinstance(image_raw, Image.Image):
                    image = image_raw
                elif isinstance(image_raw, dict) and 'bytes' in image_raw:
                    image = Image.open(io.BytesIO(image_raw['bytes'])).convert('RGB')
                else:
                    image = Image.open(image_raw).convert('RGB')

                pixel_values = self._process_image(image)

                return dict(
                    pixel_values=pixel_values,
                    type='text2image',
                    text=sample['text'],
                    image_dir=None,
                    image_file=str(sample.get('image_raw', 'stream'))
                )
            except Exception as e:
                last_exception = e
                if self.debug:
                    print(f'[HFStreamingT2IDataset] retry {retry}: {e}', flush=True)
                time.sleep(1.0)
                continue

        raise RuntimeError(f'[HFStreamingT2IDataset] Failed to fetch valid sample after 10 retries. Last error: {repr(last_exception)}')


class HFStreamingEditingDataset(Dataset):
    """Map-style adapter over Hugging Face streaming image editing datasets.

    Streams editing triplets: (source_image, instruction, target_image) directly
    without local disk caching.
    """

    def __init__(self,
                 dataset_name,
                 split='train',
                 config_name=None,
                 source_image_key='source_image',
                 target_image_key='target_image',
                 instruction_key='instruction',
                 image_process='fix_pixels',
                 image_size=512,
                 unit_image_size=32,
                 virtual_size=200000,
                 request_timeout=15,
                 debug=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.source_image_key = source_image_key
        self.target_image_key = target_image_key
        self.instruction_key = instruction_key
        self.image_process = image_process
        self.image_size = image_size
        self.unit_image_size = unit_image_size
        self.virtual_size = int(virtual_size)
        self.request_timeout = request_timeout
        self.debug = debug
        self._stream = None
        self._iter = None

    def _build_stream(self):
        if self.config_name is None:
            self._stream = load_dataset(self.dataset_name, split=self.split, streaming=True)
        else:
            self._stream = load_dataset(self.dataset_name, self.config_name, split=self.split, streaming=True)
        self._iter = iter(self._stream)

    def _get_iter(self):
        if getattr(self, '_iter', None) is None:
            self._build_stream()
        return self._iter

    def __len__(self):
        return self.virtual_size

    def _process_image(self, image):
        if not isinstance(image, Image.Image):
            if isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image)).convert('RGB')
            else:
                raise TypeError(f"Expected PIL Image or bytes, got {type(image)}")
        else:
            image = image.convert('RGB')

        if self.image_process == 'crop2square':
            image = crop2square(image)
            image = image.resize(size=(self.image_size, self.image_size))
        elif self.image_process == 'dynamic':
            image = resize_image_dynamic(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'fix_pixels':
            image = resize_image_fix_pixels(x=image, image_size=self.image_size, unit_image_size=self.unit_image_size)
        elif self.image_process == 'resize2square':
            image = image.resize(size=(self.image_size, self.image_size))
        else:
            raise NotImplementedError(f'Unsupported image_process: {self.image_process}')

        assert image.width % self.unit_image_size == 0
        assert image.height % self.unit_image_size == 0

        pixel_values = torch.from_numpy(np.array(image)).float()
        pixel_values = pixel_values / 255
        pixel_values = 2 * pixel_values - 1
        pixel_values = rearrange(pixel_values, 'h w c -> c h w')
        return pixel_values

    def _load_image_object(self, image_raw):
        if isinstance(image_raw, str) and (image_raw.startswith('http://') or image_raw.startswith('https://')):
            response = requests.get(image_raw, timeout=self.request_timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')
        elif isinstance(image_raw, Image.Image):
            return image_raw.convert('RGB')
        elif isinstance(image_raw, dict) and 'bytes' in image_raw:
            return Image.open(io.BytesIO(image_raw['bytes'])).convert('RGB')
        elif isinstance(image_raw, (bytes, bytearray)):
            return Image.open(io.BytesIO(image_raw)).convert('RGB')
        else:
            return Image.open(image_raw).convert('RGB')

    def _next_metadata(self):
        it = self._get_iter()
        for _ in range(50):
            try:
                sample = next(it)
            except StopIteration:
                self._build_stream()
                it = self._get_iter()
                sample = next(it)
            except Exception as e:
                import time
                if self.debug:
                    print(f"[_next_metadata] Error during next(it): {e}. Rebuilding stream...", flush=True)
                time.sleep(2.0)
                self._build_stream()
                it = self._get_iter()
                sample = next(it)

            # Resolve instruction text
            instruction = sample.get(self.instruction_key)
            if instruction is None:
                for k in ['edit_instruction', 'instruction', 'prompt', 'edit_prompt', 'text', 'caption']:
                    if k in sample and sample[k]:
                        instruction = sample[k]
                        break

            # Resolve source and target images
            src_raw = sample.get(self.source_image_key) or sample.get('source_image') or sample.get('image') or sample.get('input_image')
            tgt_raw = sample.get(self.target_image_key) or sample.get('edited_image') or sample.get('target_image') or sample.get('output_image')

            if instruction and src_raw is not None and tgt_raw is not None:
                return dict(instruction=str(instruction).strip(), src_raw=src_raw, tgt_raw=tgt_raw)

        raise RuntimeError(
            f'Failed to get valid editing sample from {self.dataset_name}.'
        )

    def __getitem__(self, idx):
        import time
        last_exception = None
        for retry in range(10):
            try:
                sample = self._next_metadata()
                src_img = self._load_image_object(sample['src_raw'])
                tgt_img = self._load_image_object(sample['tgt_raw'])

                pixel_values_src = self._process_image(src_img)
                pixel_values_tgt = self._process_image(tgt_img)

                return dict(
                    pixel_values=pixel_values_tgt,
                    pixel_values_src=[pixel_values_src],
                    type='image2image',
                    text=sample['instruction'],
                    image_dir=None,
                    image_file="stream_edit"
                )
            except Exception as e:
                last_exception = e
                if self.debug:
                    print(f'[HFStreamingEditingDataset] retry {retry}: {e}', flush=True)
                time.sleep(1.0)
                continue

        raise RuntimeError(f'[HFStreamingEditingDataset] Failed to fetch valid sample after 10 retries. Last error: {repr(last_exception)}')


class HFStreamingJointDataset(Dataset):
    """Joint dataset adapter dynamically interleaving T2I and Image Editing streams.
    """

    def __init__(self,
                 t2i_dataset,
                 edit_dataset,
                 t2i_ratio=0.5,
                 virtual_size=400000):
        super().__init__()
        from xtuner.registry import BUILDER
        if isinstance(t2i_dataset, dict):
            self.t2i_dataset = BUILDER.build(t2i_dataset)
        else:
            self.t2i_dataset = t2i_dataset

        if isinstance(edit_dataset, dict):
            self.edit_dataset = BUILDER.build(edit_dataset)
        else:
            self.edit_dataset = edit_dataset

        self.t2i_ratio = float(t2i_ratio)
        self.virtual_size = int(virtual_size)

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, idx):
        if random.random() < self.t2i_ratio:
            return self.t2i_dataset[idx]
        else:
            return self.edit_dataset[idx]

