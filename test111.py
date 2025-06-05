import sys
import os
import argparse
import yaml
import json
import time
import asyncio
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
import copy
from typing import List,Union,Literal
from datasets.gsm8k_dataset import gsm_data_process,gsm_get_predict,svamp_data_process, multiarith_data_process
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
from typing import Iterator
from AgentDropout.graph.graph import Graph
from AgentDropout.tools.reader.readers import JSONLReader
from AgentDropout.tools.coding.python_executor import PyExecutor
from AgentDropout.utils.globals import Time
from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentDropout.utils.utils import nuclear_norm,frobenius_norm
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from experiments.ddc import calculate_D
def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)

    

async def main():
    result_file = None
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        while True:
            for idx in range(len(dataset_train)):
                record = dataset_train[idx]
                yield record
    #dataset = JSONLReader.parse_file("datasets/humaneval/humaneval-py.jsonl")
    #train_dataset = JSONLReader.parse_file('datasets/gsm8k/train.jsonl')
    #train_dataset = gsm_data_process(train_dataset)
    result_dir = Path(f"{AgentPrune_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    download()
    dataset_train = MMLUDataset('dev')
    loader = infinite_data_loader()
    for i_record, record in zip(range(280), loader):
            #current_batch = dataloader(dataset_train,20,i_batch)
            #for i_record, record in enumerate(current_batch):
                sample_input_dict = dataset_train.record_to_input(record)
                sample_task = sample_input_dict["task"]
                D = calculate_D(sample_task)
                print(D)





if __name__ == '__main__':
    asyncio.run(main())
