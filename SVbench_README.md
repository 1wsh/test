## Install

```
cd /InternVL/internvl_chat 

pip install -r internal_chat.txt
```

## Data Preapration

Data download reference [https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html]()

## Evaluation

### Image Datasets:

### MMBench

```
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge mmbench-test-en --dynamic
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are: 

```
overall: 80.66
```

### CCBench

```
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge ccbench-dev --dynamic
```

Then, submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission). The expected test results are: 

```
overall:74.71
```

###  Tiny LVLM

```
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge tiny_lvlm --dynamic
```

The expected test results are:

```
  Visual_Perception: 0.4825
  ObjecCHallucination: 0.9033333333333333
  Visual_Commonsense: 0.636
  Visual_Knowledge_Acquisition: 0.6842857142857143
  Visual_Reasoning: 0.6654545454545454
  Overall: 3.371573593073593
```

### MM-Vet

```
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge mmvet --dynamic
```

Then, submit the results to the [evaluation server](https://huggingface.co/spaces/whyu/MM-Vet_Evaluator). The expected test results are:

```
runs:36.9
```

### MMMU

```
GPUS=8 bash evaluate.sh work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge mmmu-val --dynamic
```

The expected test results are:

```
{'Overall-Art and Design': {'num': 120, 'acc': 0.592}, 'Art': {'num': 30, 'acc': 0.733}, 'Art_Theory': {'num': 30, 'acc': 0.6},  'Overall': {'num': 900, 'acc': 0.49}}
```

## Video Datasets:

### MMBench Video

```
conda create -n mmvideo python=3.8.10
cd VLMEvalKit
pip install -e .
```

Other configuration information reference [MMBench VIdeo Configiguration](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/get_started/Quickstart.md)

```
torchrun --nproc-per-node=8 run.py --data MMBench-Video --model InternVL2-8B --verbose --nframe 8
```

The expected test results are:

```
 "coarse_all": {
        "CP": "1.53",
        "FP-S": "1.41",
        "FP-C": "1.16",
        "HL": "0.21",
        "LR": "1.06",
        "AR": "1.55",
        "RR": "1.59",
        "CSR": "1.37",
        "TR": "1.31",
        "Perception": "1.35",
        "Reasoning": "1.39",
        "Overall": "1.37"
    },
    "coarse_valid": {
        "CP": "1.53",
        "FP-S": "1.41",
        "FP-C": "1.16",
        "HL": "0.21",
        "LR": "1.06",
        "AR": "1.55",
        "RR": "1.59",
        "CSR": "1.37",
        "TR": "1.31",
        "Perception": "1.35",
        "Reasoning": "1.39",
        "Overall": "1.37"
    }
}
```



### Video MME

```
GPUS=8 bash evaluate.sh /hetu_group/huangkangwei/InternVL/internvl_chat/work_dirs/internvl_chat_v2_0/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_yzy_v6_merge  videomme --dynamic --max-num 1 --out-tag svbenchmark_v6_merge
```

