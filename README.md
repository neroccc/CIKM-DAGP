# DAGP

DAGP: Difficulty-Aware Graph Pruning for Multi-Agent Reasoning with Large Language Models


#
1. **Environment Setup**:

```shell
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

2. **API Configuration**:

```python
# Update in DAPG/llm/gpt_chat.py
MINE_BASE_URL = ""
MINE_API_KEYS = ""
```

3. **Local Model Deployment** (Optional):

```bash
# Using vLLM for local inference
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/model --dtype auto --api-key API_KEYS --port 5000
```

```python
api_key = API_KEYS
base_url = "http://localhost:5000/v1"
```


# Quick Start

Run DAGP on GSM8K (other datasets are similar): 

python experiments/run_gsm8k.py \
  --agent_nums 5 \
  --mode FullConnected \
  --batch_size 40 \
  --num_iterations 2 \
  --imp_per_iterations 1 \
  --pruning_rate 0.10 \
  --num_rounds 2 \
  --llm_name /data/models/Meta-Llama-3-8B-Instruct \
  --optimized_spatial \
  --optimized_temporal \
  --diff \
  --dec




#Acknowledgments

Code framework based on [GPTSwarm](https://github.com/metauto-ai/GPTSwarm), [AgentPrune](https://github.com/yanweiyue/AgentPrune) and [AgentDropout](https://github.com/wangzx1219/AgentDropout).
