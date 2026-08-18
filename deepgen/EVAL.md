## Evaluation
For now we only provide inference code, please turn to the official repos of the benchmarks to calculate final performance.
the `/path/to/your/ckpt` can be both `.pth folder` or `.pt file`

### GenEval
- See [GenEval](https://github.com/djghosh13/geneval/tree/main) for original GenEval prompts and put in evaluation/geneval
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/gen_eval.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/geneval/geneval_prompt.jsonl \
         --height 512 \
         --width 512 \
         --seed 42
```

### DPGBench
- See [DPGBench](https://github.com/TencentQQGYLab/ELLA) for original DPGBench prompts and put in evaluation/DPGBench
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/dpg_bench.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/DPG-Bench/prompts \
         --height 512 \
         --width 512 \
         --seed 42
```

### UniGenBench
- Please download the benchmark data from [UniGenBench](https://github.com/CodeGoat24/UniGenBench/blob/main/data/test_prompts_en.csv) and place it in the evaluation/UniGenBench.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/unigenbench.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/UniGenBench/test_prompts_en.csv \
         --height 512 \
         --width 512 \
         --seed 42
```
### WISE
- Please download the benchmark data from [WISE](https://github.com/PKU-YuanGroup/WISE/tree/main/data) and place it in the evaluation/WISE.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/wise.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/wise/data/spatio-temporal_reasoning.json \ # for spatio-temporal domain
         --height 512 \
         --width 512 \
         --seed 42
```

### T2I-CoREBench
- Please download the benchmark data from [T2I-CoreBench](https://huggingface.co/datasets/lioooox/T2I-CoReBench) and place it in the evaluation/T2I-CoReBench-main.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/corebench.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/T2I-CoReBench-main/corebench.jsonl \
         --height 512 \
         --width 512 \
         --seed 42
```

### ImgEdit
- Please download the benchmark data from [ImgEdit-Bench](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar) and place it in the evaluation/ImgEdit.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/img_edit.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/ImgEdit/Benchmark/singleturn \ 
         --height 512 \
         --width 512 \
         --seed 42
```

### GEdit
- Please download the benchmark data from [GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench) and place it in the evaluation/GEdit-Bench.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/gedit.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/GEdit-Bench \ 
         --height 512 \
         --width 512 \
         --seed 42
```

### RISE
- Please download the benchmark data from [RISE](https://huggingface.co/datasets/PhoenixZ/RISEBench) and place it in the evaluation/RISEBench-full.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/rise_bench.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/RISEBench-full \ 
         --height 512 \
         --width 512 \
         --seed 42
```

### UniREditBench
- Please download the benchmark data from [UniREditBench](https://maplebb.github.io/UniREditBench/) and place it in the evaluation/UniREditBench.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/unireditbench.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/UniREditBench \ 
         --height 512 \
         --width 512 \
         --seed 42
```

### CVTG
- Please download the benchmark data from [CVTG](https://github.com/NJU-PCALab/TextCrafter) and place it in the evaluation/CVTG-2K.
```shell
export PYTHONPATH=.
accelerate launch scripts/evaluation/CVTG.py  \ 
         --checkpoint /path/to/your/ckpt   \ 
         --output /path/to/save/results \
         --data evaluation/CVTG-2K \ 
         --height 512 \
         --width 512 \
         --seed 42
```
