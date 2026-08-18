# Data prepration

## Pretrain t2i
the Pretrain t2i dataset config can be found in **`configs/datasets/deepgen_512_fix_pixels/t2i_pretrain.py`**. We use [OpenUni](https://github.com/wusize/OpenUni/blob/main/docs/DATASETS.md)  as our pretrain t2i dataset and following data process steps.
Please modify `data_path` and `image_folder` as your local path

## SFT t2i
the Pretrain t2i dataset config can be found in **`configs/datasets/deepgen_512_fix_pixels/t2i_sft_zh.py`**.
Please modify `data_path` and `image_folder` as your local path
The open-source T2I SFT datasets we used are below:
| DATASET        | Download Link                                                |
| ---------- | ------------------------------------------------------------ |
| BLIP3o-60k | https://huggingface.co/datasets/BLIP3o/BLIP3o-60k |
| ShareGPT-4o-Image-T2I | https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image |
| OpenGPT-4o-Image-T2I | https://huggingface.co/datasets/WINDop/OpenGPT-4o-Image |
| Echo-4o-Image | https://huggingface.co/datasets/Yejy53/Echo-4o-Image |
| UniReason-T2I | https://huggingface.co/datasets/Alex11556666/Reason_Tuning |

the banana-50k and text rendering data we will upload in[DeepGen Data](https://huggingface.co/datasets/DeepGenTeam/DeepGen-1.0) soon.

We have refactored the dataloader code so that data for any task can be passed through JSON files. After downloading the dataset, ensure that the text prompts are processed into JSON format, the specific format is as follows:

T2I SFT data format JSON file:
```
.....
  {
    "type": "T2I_SFT",
    "txt": "Waist-up, a male figure with jet-black hair in cyberpunk-80s attire, his form defined by Renaissance chiaroscuro, intently deciphers hieroglyphs on a massive column under vibrant top-down golden hour light, the column leading the eye within a plein air composition.",
    "image": "",
    "image_path": "image/16685.png"
  },
.....
```

## Edit 
the Pretrain edit dataset config can be found in **`configs/datasets/deepgen_512_fix_pixels/edit_pretrain.py`**.
the SFT edit dataset config can be found in **`configs/datasets/deepgen_512_fix_pixels/edit_sft_zh.py`**.
Please modify `data_path` and `image_folder` as your local path
The open-source Editing datasets we used are below:
| DATASET        | Download Link                                                |
| ---------- | ------------------------------------------------------------ |
| Nano-consistent-150k |https://huggingface.co/datasets/Yejy53/Nano-consistent-150k |
| ShareGPT-4o-Image-Editing | https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-4o-Image |
| OpenGPT-4o-Image-Editing | https://huggingface.co/datasets/WINDop/OpenGPT-4o-Image |
| pico-banana-400k | https://github.com/apple/pico-banana-400k |
| OmniGen2-X2I2 | https://huggingface.co/datasets/OmniGen2/X2I2 |
| UniWorld-V1-Edit set | https://huggingface.co/datasets/LanguageBind/UniWorld-V1 |
| NHR-edit | https://huggingface.co/datasets/iitolstykh/NHR-Edit |
| NHR-edit-part 2 | https://huggingface.co/datasets/iitolstykh/NHR-Edit-part2 |
| UniReason-Edit | https://huggingface.co/datasets/Alex11556666/Reason_Tuning |


We have refactored the dataloader code so that data for any task can be passed through JSON files. After downloading the dataset, ensure that the text prompts are processed into JSON format, the specific format is as follows:

Editing data format JSON file:
```
.....
  {
    "input_image": [
      "editing/input_11656.jpg"
    ],
    "output_image": "editing/output_11656.png",
    "instruction": "Change the image style to Monet's Impressionist style."
  },
.....
```


1. **Download the dataset from link** Follow the official process for each data to extract, then process it into the above JSON file format.

2. Edit every `data_path` and `image_folder` placeholder in dataset config .
   
3. the t2i and edit joint training dataset config can be found in **`configs/datasets/deepgen_512_fix_pixels/joint_pretrain.py`** and **`configs/datasets/deepgen_512_fix_pixels/joint_sft_zh.py`**, Edit`batch_sizes` and `repeats = [1, 2]`(the training radio between editing and generation task) .
