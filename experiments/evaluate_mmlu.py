import os
import json
import math
import time
import asyncio
from typing import Union,Literal,Optional,Iterator,List,Any,Dict
from tqdm import tqdm
import copy
import time
from AgentDropout.utils.globals import Time
from pathlib import Path
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentDropout.utils.const import AgentPrune_ROOT
from AgentDropout.graph.graph import Graph
from experiments.accuracy import Accuracy
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens
from ddc import calculate_D
async def evaluate(
        saved_spatial_masks, 
        saved_temporal_masks,
        graph:Graph,
        dataset,
        num_rounds:int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 4,
        dec: bool = False,
        args=None,
        ) -> float:

    print(f"Evaluating AgentDropout on {dataset.__class__.__name__} split {dataset.split}")
    
    if not graph.diff:
        graph.spatial_logits.requires_grad_(False)
        graph.temporal_logits.requires_grad_(False)
    else:
        for i in range(graph.rounds):
            graph.base_spatial_logits[i].requires_grad_(False)
        for i in range(graph.rounds - 1):
            graph.base_temporal_logits[i].requires_grad_(False)
    
    accuracy = Accuracy()
    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None:
                if i_record >= limit_questions:
                    break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if len(records) > 0:
            yield records
        return
    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    data=[]
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    result_dir = Path(f"{AgentPrune_ROOT}/result/mmlu")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_llama3_{current_time}.json"
    grand_spatial = 0
    grand_temporal = 0
    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        print(80*'-')

        start_ts = time.time()
        answer_log_probs = []
        
        for record in record_batch:
            sample_input_dict = dataset.record_to_input(record)
            sample_task = sample_input_dict["task"]
            D = calculate_D(sample_task)
            realized_graph = copy.deepcopy(graph)
            realized_graph.set_difficulty_tensor(torch.tensor(D, dtype=torch.float32))
            realized_graph.adjust_graph_structure(D)
            for i in range(graph.rounds):
                realized_graph.spatial_masks[i].data = saved_spatial_masks[i].clone()
            for i in range(graph.rounds - 1):
                realized_graph.temporal_masks[i].data = saved_temporal_masks[i].clone()
            if args.diff:
                realized_graph.refine_masks_from_existing(realized_graph.dynamic_pruning_rate)
            else:
                realized_graph.update_masks(realized_graph.dynamic_pruning_rate)             
            #realized_graph.spatial_logits = graph.spatial_logits
            #realized_graph.temporal_logits = graph.temporal_logits
            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds,case=True)))
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs, all_answers, sp_per_round, tp_per_round, edge_counts = zip(*raw_results)
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        for raw_answer, record, all_answer in zip(raw_answers, record_batch, all_answers):
            print("Raw answer:", raw_answer)
            answer = dataset.postprocess_answer(raw_answer)
            print("Postprocessed answer:", answer)
            correct_answer = dataset.record_to_target_answer(record)
            print("Correct answer:", correct_answer)
            accuracy.update(answer, correct_answer)
            accuracy.print()
            updated_item = {
                "Question": dataset.record_to_input(record)['task'],
                "Answer": correct_answer,
                "All_answers": all_answer,
                "Response": raw_answer,
            }
            data.append(updated_item)
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
    accuracy.print()
    print("Done!")

    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
