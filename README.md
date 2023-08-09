# GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution

* paper: https://arxiv.org/abs/2307.08775

## Requirements
```
conda create --name GEAR python=3.7.10
conda activate GEAR
bash install.sh
```

## What does data looks like
The sampled datasets used in our experiments can be found in ```/datasets```.

```/gear/read_dataset.py``` provides functions for reading those dataset files.

## Instructions
Add OpenAI Api key in ```api.py``` and ```OpenAIModels.py```

Args Explanation
```
parser.add_argument("--slm1", help="small language model",
                    type=str, default="EleutherAI/gpt-neo-1.3B")
parser.add_argument("--slm2", help="small language model2",
                    type=str, default="sentence-transformers/all-mpnet-base-v2")
parser.add_argument("-v", "--verbose", help="verbose",
                    action="store_true")
parser.add_argument("-M", "--max_tokens", help="max tokens",
                     type=int, default=512)
parser.add_argument("-t", "--top_k", help="Return top k APIs",
                    type=int, default=1)
parser.add_argument("--llm", help="large language model",
                    type=str, default="EleutherAI/gpt-j-6B")
parser.add_argument("-e", "--early_stop", help="early stop the program", 
                    type=int, default = 0)
parser.add_argument("-c", "--check_point", help="checkpoint file path", 
                    type=str, default=None)
parser.add_argument("-d", "--dataset", help="dataset path",
                    type=str)
parser.add_argument("-o", "--output", help="output file path",
                    type=str, default=None)
parser.add_argument('--experiment', choices=['gptj', 'gptj_zero_shot', 'openai', "gptj_few_shot", "openai_zero_shot", "openai_few_shot", "ground"], nargs="+")
parser.add_argument('--openai_model', choices=['chatgpt', 'gpt3'], default="gpt3")
parser.add_argument("--tool", choices=["calculator", "wiki", "qa", "mt", "multilingualqa", "timezone", "sleep", "log", "exp", "robot"], nargs="+")
parser.add_argument('--prompt', choices=['mt', 'wiki', 'qa', 'calculator'], default="mt")
parser.add_argument('--fdevice', type=str, default="cuda:0")
parser.add_argument('--ALPHA', type=float, default=0.75)
```

Run GEAR on GPT-J with four basic tools
```
cd gear
python -u main.py -v -t 1 -e 1000 \
-d {DATASET_PATH} \
-c {CHECKPOINT_JSON_PATH} \
-o {OUTPUT_JSON_PATH} \
--experiment gptj \
--tool calculator wiki qa mt \
--fdevice {DEVICE} \
> {OUTPUT_TXT_PATH}
```

Run GEAR on GPT-3 Model with four basic tools
```
cd gear
python -u main.py -v -t 1 -e 1000 \
-d {DATASET_PATH} \
-c {CHECKPOINT_JSON_PATH} \
-o {OUTPUT_JSON_PATH} \
--experiment openai \
--tool calculator wiki qa mt \
--openai_model gpt3 \
--fdevice {DEVICE} \
> {OUTPUT_TXT_PATH}
```
OpenAI model name can be changed to ```chatgpt```

Run zero-shot and few-shot experiments for GPT-J and GPT-3 with four basic tools
```
cd gear
python -u main.py -v -t 1 -e 1000 \
-d {DATASET_PATH} \
-c {CHECKPOINT_JSON_PATH} \
-o {OUTPUT_JSON_PATH} \
--experiment gptj_zero_shot gptj_few_shot openai_zero_shot openai_few_shot \
--prompt {TASK_NAME} \
--tool calculator wiki qa mt \
--openai_model gpt3 \
--fdevice {DEVICE} \
> {OUTPUT_TXT_PATH}
```

## FAQ
**The Program is running so slow and WikiSearch returns nothing.** This is because of the AWS connection issue from the URL used in the Wikipedia Search tool. Our tests conducted between April and June 2023 show that the server works well and typically takes 2-3 seconds to return a result, yet after June 20th, retrieval times increased to 120 seconds without returning anything. One potential solution here is to change the Wikisearch URL or use the Python [Wikipedia Search package](https://pypi.org/project/wikipedia/) until it is fixed, but may not guarantee the same experiment result. 

Reach out to Yining ylu130@jh.edu or Haoping hyu90@jh.edu if you have any other questions! :)

## How to Cite
```bibtex
@article{lu2023gear,
  title={GEAR: Augmenting Language Models with Generalizable and Efficient Tool Resolution},
  author={Lu, Yining and Yu, Haoping and Khashabi, Daniel},
  journal={arXiv preprint arXiv:2307.08775},
  year={2023}
}
```
