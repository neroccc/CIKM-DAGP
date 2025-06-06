import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import torch.nn.functional as F
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from DAGP.graph.graph import Graph
from DAGP.tools.reader.readers import JSONLReader
from DAGP.tools.coding.python_executor import PyExecutor
from DAGP.utils.globals import Time
from DAGP.utils.const import AgentPrune_ROOT
from DAGP.utils.globals import Cost, PromptTokens, CompletionTokens
from DAGP.utils.utils import nuclear_norm,frobenius_norm
from ddc import calculate_D
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
    
def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on HumanEval")
    parser.add_argument("--dataset_json", type=str, default="datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4-1106-preview")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--delta', type=float, default=0.1, help="noise level")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5,help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=2,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default = 10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="humaneval",help="Domain (the same as dataset name), default 'humaneval'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalWriteCode',
                        help='The decison method of the agentprune')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    parser.add_argument('--diff',action='store_true')
    parser.add_argument('--dec',action='store_true')
    parser.add_argument('--cot',action='store_true')
    
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

async def main():
    args = parse_args()
    result_file = None
    dataset = JSONLReader.parse_file(args.dataset_json)
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/eval")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_llama3_{current_time}.json"
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    decision_method = args.decision_method
    kwargs = get_kwargs(args.mode,len(agent_names))

    graph = Graph(domain="humaneval",
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  rounds=args.num_rounds,
                  diff=args.diff,
                  dec=args.dec,
                  **kwargs)
    controller_params = [
            graph.controller_weights,
            graph.controller_bias,
            graph.sparsity_base,
            graph.sparsity_delta1,
            graph.sparsity_delta2
        ]

    if args.dec:
        graph.optimized_spatial=False
        graph.optimized_temporal=False
        total_solved, total_executed = (0, 0)


        if not graph.diff:
            optimizer = torch.optim.Adam(
                [graph.spatial_logits_1, graph.temporal_logits_1] + controller_params,
                lr=args.lr
            )
        else:
            optimizer = torch.optim.Adam(
                list(graph.spatial_logits_1.parameters()) +
                list(graph.temporal_logits_1.parameters()) +
                controller_params,
                lr=args.lr
            )        



        for i_batch in range(2):
            print(f"dec Batch {i_batch}",80*'-')
            start_ts = time.time()
            answer_log_probs = []
            tests = []
            add_losses = []        
            current_batch = dataloader(dataset,20,i_batch)
            if current_batch is None:
                print("No more data available.")
                break
            
            for i_record, record in enumerate(current_batch):
                task = record["prompt"]
                D = calculate_D(task)

                realized_graph = copy.deepcopy(graph)
                realized_graph.spatial_logits_1 = graph.spatial_logits_1
                realized_graph.temporal_logits_1 = graph.temporal_logits_1
                realized_graph.set_difficulty_tensor(torch.tensor(D, dtype=torch.float32))
                
                if not graph.diff:
                    spatial_matrix_train = realized_graph.spatial_logits_1.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                    temporal_matrix_train = realized_graph.temporal_logits_1.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                else:
                    spatial_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.spatial_logits_1]
                    temporal_matrix_train = [param.reshape((sum(args.agent_nums), sum(args.agent_nums))) for param in realized_graph.temporal_logits_1]

                spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
                temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
                if not graph.diff:
                    loss_s = nuclear_norm(spatial_matrix_train)
                    loss_t = nuclear_norm(temporal_matrix_train)
                    frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                    frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                else:
                    loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                    loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                    frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                    frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
                sparsity_weight = graph.get_sparsity_weight(D)
                add_loss = sparsity_weight * (loss_s + loss_t) + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
                add_loss = 0
                task = record["prompt"]
                test = record["test"]
                tests.append(test)
                input_dict = {"task": task}
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds,skip=True)))
                add_losses.append(add_loss)
                
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs, sp_per_round, tp_per_round, edge_counts = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            data = load_result(result_file)
                
            for task, answer, log_prob, add_loss, test in zip(current_batch, raw_answers, log_probs, add_losses, tests):
                if not isinstance(answer,list):
                    raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
                answer = answer[0].lstrip("```python\n").rstrip("\n```")
                is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=5)
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                accuracy = total_solved/ total_executed
                utility = is_solved
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss+add_loss)
                updated_item = {
                    "Question": task,
                    "Tests": test,
                    "Attempt answer": answer,
                    "Solved": is_solved,
                    "Solution": answer,
                    "Total solved": total_solved,
                    "Total executed": total_executed,
                    "Accuracy": accuracy
                }
                data.append(updated_item)

            with open(result_file, 'w',encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if not graph.diff:
                spatial_probs = torch.sigmoid(graph.spatial_logits_1)
                temporal_probs = torch.sigmoid(graph.temporal_logits_1)
            else:
                spatial_probs = [torch.sigmoid(logit) for logit in graph.spatial_logits_1]
                temporal_probs = [torch.sigmoid(logit) for logit in graph.temporal_logits_1]
            
            print(f"Batch time {time.time() - start_ts:.3f}")
            print(f"Accuracy: {accuracy}")
            print("utilities:", utilities)
            print("loss:", total_loss.item())
            print("Spatial logits:", graph.spatial_logits_1)
            print("Temporal logits:", graph.temporal_logits_1)
            print("Spatial probs:", spatial_probs)
            print("Temporal probs:", temporal_probs)
            print("Spatial masks:", graph.spatial_masks)
            print("Temporal logits:", graph.temporal_masks)
            
            spatial_mask_values = [param.data.tolist() for param in graph.spatial_masks]
            print("Spatial masks as list of lists:", spatial_mask_values)
            temporal_mask_values = [param.data.tolist() for param in graph.temporal_masks]
            print("Temporal masks as list of lists:", temporal_mask_values)
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")
        if isinstance(graph.spatial_masks, torch.nn.ParameterList):
            old_spatial = [mask.detach().clone() for mask in graph.spatial_masks]
            old_temporal = [mask.detach().clone() for mask in graph.temporal_masks]
        else:
            old_spatial = graph.spatial_masks.detach().clone()
            old_temporal = graph.temporal_masks.detach().clone()
        graph.update_masks_dec()
        saved_spatial_masks = [mask.detach().clone() for mask in graph.spatial_masks]
        saved_temporal_masks = [mask.detach().clone() for mask in graph.temporal_masks]
        # 3. 获取更新后的新 masks
        if isinstance(graph.spatial_masks, torch.nn.ParameterList):
            new_spatial = [mask for mask in graph.spatial_masks]
            new_temporal = [mask for mask in graph.temporal_masks]
        else:
            new_spatial = graph.spatial_masks
            new_temporal = graph.temporal_masks

        # 4. 找出哪些边被从 1 → 0
        def diff_round(old_masks, new_masks, potential_edges, kind="spatial"):
            removed = []
            for i, (old_m, new_m) in enumerate(zip(old_masks, new_masks)):
                # old_m 和 new_m 都是 shape [E] 的 tensor
                # 找到 old_m==1 且 new_m==0 的位置
                idxs = ((old_m == 1) & (new_m == 0)).nonzero(as_tuple=False).flatten().tolist()
                for edge_idx in idxs:
                    u, v = potential_edges[edge_idx]
                    removed.append((i, u, v))
            return removed

        removed_spatial = diff_round(old_spatial, new_spatial, graph.potential_spatial_edges, "spatial")
        removed_temporal = diff_round(old_temporal, new_temporal, graph.potential_temporal_edges, "temporal")

        # 5. 打印结果
        for round_i, u, v in removed_spatial:
            print(f"[DEC] Round {round_i} 删除了 spatial 边：{u} -> {v}")
        for round_i, u, v in removed_temporal:
            print(f"[DEC] Round {round_i} 删除了 temporal 边：{u} -> {v}")


    if not graph.diff:
        optimizer = torch.optim.Adam(
            [graph.spatial_logits, graph.temporal_logits] + controller_params,
            lr=args.lr
        )
    else:
        optimizer = torch.optim.Adam(
            list(graph.base_spatial_logits.parameters()) +
            list(graph.base_temporal_logits.parameters()) +
            list(graph.spatial_logit_mlps.parameters()) +
            list(graph.temporal_logit_mlps.parameters()) +
            controller_params,
            lr=args.lr
        )
    num_batches = int(len(dataset)/args.batch_size)
    total_solved, total_executed = (0, 0)

    if args.optimized_temporal or args.optimized_spatial:
        graph.optimized_spatial=True
        graph.optimized_temporal=True

        for i_batch in range(4):
            print(f"train Batch {i_batch}",80*'-')
            start_ts = time.time()
            answer_log_probs = []
            tests = []
            add_losses = []        
            current_batch = dataloader(dataset,10,i_batch)
            if current_batch is None:
                print("No more data available.")
                break
            
            for i_record, record in enumerate(current_batch):
                task = record["prompt"]
                D = calculate_D(task)
                graph.set_difficulty_tensor(torch.tensor(D, dtype=torch.float32))
                realized_graph = copy.deepcopy(graph)
                if not graph.diff:
                    spatial_matrix_train = realized_graph.base_spatial_logits.reshape((sum(args.agent_nums), sum(args.agent_nums)))
                    temporal_matrix_train = realized_graph.base_temporal_logits.reshape((sum(args.agent_nums), sum(args.agent_nums)))
                else:
                    spatial_matrix_train = [
                        (realized_graph.base_spatial_logits[i] + realized_graph.spatial_logit_mlps[i](graph.current_difficulty_tensor)).reshape((sum(args.agent_nums), sum(args.agent_nums)))
                        for i in range(args.num_rounds)
                    ]
                    temporal_matrix_train = [
                        (realized_graph.base_temporal_logits[i] + realized_graph.temporal_logit_mlps[i](graph.current_difficulty_tensor)).reshape((sum(args.agent_nums), sum(args.agent_nums)))
                        for i in range(args.num_rounds - 1)
                    ]
                spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
                temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
                if not graph.diff:
                    loss_s = nuclear_norm(spatial_matrix_train)
                    loss_t = nuclear_norm(temporal_matrix_train)
                    frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                    frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                else:
                    loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                    loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                    frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                    frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
                sparsity_weight = graph.get_sparsity_weight(D)
                add_loss = sparsity_weight * (loss_s + loss_t) + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
                
                task = record["prompt"]
                graph.adjust_graph_structure(D)
                test = record["test"]
                tests.append(test)
                input_dict = {"task": task}
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
                add_losses.append(add_loss)
                
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs, sp_per_round, tp_per_round, edge_counts = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            data = load_result(result_file)
                
            for task, answer, log_prob, add_loss, test in zip(current_batch, raw_answers, log_probs, add_losses, tests):
                if not isinstance(answer,list):
                    raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
                answer = answer[0].lstrip("```python\n").rstrip("\n```")
                is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                accuracy = total_solved/ total_executed
                utility = is_solved
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss+add_loss)
                updated_item = {
                    "Question": task,
                    "Tests": test,
                    "Attempt answer": answer,
                    "Solved": is_solved,
                    "Solution": answer,
                    "Total solved": total_solved,
                    "Total executed": total_executed,
                    "Accuracy": accuracy
                }
                data.append(updated_item)
            with open(result_file, 'w',encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if not graph.diff:
                spatial_probs = torch.sigmoid(graph.spatial_logits)
                temporal_probs = torch.sigmoid(graph.temporal_logits)
            else:
                spatial_probs = [
                    torch.sigmoid(graph.base_spatial_logits[i] + graph.spatial_logit_mlps[i](graph.current_difficulty_tensor))
                    for i in range(args.num_rounds)
                ]
                temporal_probs = [
                    torch.sigmoid(graph.base_temporal_logits[i] + graph.temporal_logit_mlps[i](graph.current_difficulty_tensor))
                    for i in range(args.num_rounds - 1)
                ]
            
            print(f"Batch time {time.time() - start_ts:.3f}")
            print(f"Accuracy: {accuracy}")
            print("utilities:", utilities)
            print("loss:", total_loss.item())
            if not graph.diff:
                print("Spatial logits:", graph.base_spatial_logits)
                print("Temporal logits:", graph.base_temporal_logits)
            else:
                for i in range(args.num_rounds):
                    logits_i = graph.base_spatial_logits[i] + graph.spatial_logit_mlps[i](graph.current_difficulty_tensor)
                    print(f"Spatial logits (round {i}):", logits_i)
                for i in range(args.num_rounds - 1):
                    logits_i = graph.base_temporal_logits[i] + graph.temporal_logit_mlps[i](graph.current_difficulty_tensor)
                    print(f"Temporal logits (round {i}):", logits_i)
            print("Spatial probs:", spatial_probs)
            print("Temporal probs:", temporal_probs)
            print("Spatial masks:", graph.spatial_masks)
            print("Temporal logits:", graph.temporal_masks)
            spatial_mask_values = [param.data.tolist() for param in graph.spatial_masks]
            print("Spatial masks as list of lists:", spatial_mask_values)

            # Collect temporal mask values into a list
            temporal_mask_values = [param.data.tolist() for param in graph.temporal_masks]
            print("Temporal masks as list of lists:", temporal_mask_values)

            if (i_batch+1)%2 == 0 and i_batch < 4 and (args.optimized_spatial or args.optimized_temporal):
                if not graph.diff:
                    spatial_masks, temporal_masks = graph.update_masks(graph.dynamic_pruning_rate)
                else:
                    spatial_masks, temporal_masks = graph.update_masks_diff(graph.dynamic_pruning_rate)
                print("spatial masks:",spatial_masks)
                print("temporal masks:",temporal_masks)
                if not graph.diff:
                    print("spatial sparsity:",spatial_masks.sum()/spatial_masks.numel())
                    print("temporal sparsity:",temporal_masks.sum()/temporal_masks.numel())
                else:
                    print("spatial sparsity:",spatial_masks[0].sum()/spatial_masks[0].numel())
                    print("temporal sparsity:",temporal_masks[0].sum()/temporal_masks[0].numel())
            if i_batch+1 == args.num_iterations:
                args.optimized_spatial = False
                args.optimized_temporal = False
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")

    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()
    total_solved, total_executed = (0, 0)

    spatial_mask_values = [param.data.tolist() for param in graph.spatial_masks]
    print("Spatial masks as list of lists:", spatial_mask_values)
    temporal_mask_values = [param.data.tolist() for param in graph.temporal_masks]
    print("Temporal masks as list of lists:", temporal_mask_values)
    if not graph.diff:
        spatial_prob_values = torch.sigmoid(graph.base_spatial_logits).detach().cpu().tolist()
        temporal_prob_values = torch.sigmoid(graph.base_temporal_logits).detach().cpu().tolist()
    else:
        spatial_prob_values = [
            torch.sigmoid(graph.base_spatial_logits[i] + graph.spatial_logit_mlps[i](graph.current_difficulty_tensor)).detach().cpu().tolist()
            for i in range(args.num_rounds)
        ]
        temporal_prob_values = [
            torch.sigmoid(graph.base_temporal_logits[i] + graph.temporal_logit_mlps[i](graph.current_difficulty_tensor)).detach().cpu().tolist()
            for i in range(args.num_rounds - 1)
        ]
    print("Spatial probs as list of lists:", spatial_prob_values)
    print("Temporal probs as list of lists:", temporal_prob_values)


    grand_spatial = 0
    grand_temporal = 0

    for i_batch in range(4,17):
        print(f"Batch {i_batch}",80*'-')
        start_ts = time.time()
        answer_log_probs = []
        tests = []
        add_losses = []        
        current_batch = dataloader(dataset,10,i_batch)
        if current_batch is None:
            print("No more data available.")
            break
        
        for i_record, record in enumerate(current_batch):
            task = record["prompt"]
            D = calculate_D(task)
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
            if not graph.diff:
                spatial_matrix_train = realized_graph.base_spatial_logits.reshape((sum(args.agent_nums), sum(args.agent_nums)))
                temporal_matrix_train = realized_graph.base_temporal_logits.reshape((sum(args.agent_nums), sum(args.agent_nums)))
            else:
                spatial_matrix_train = [
                    (realized_graph.base_spatial_logits[i] + realized_graph.spatial_logit_mlps[i](realized_graph.current_difficulty_tensor)).reshape((sum(args.agent_nums), sum(args.agent_nums)))
                    for i in range(args.num_rounds)
                ]
                temporal_matrix_train = [
                    (realized_graph.base_temporal_logits[i] + realized_graph.temporal_logit_mlps[i](realized_graph.current_difficulty_tensor)).reshape((sum(args.agent_nums), sum(args.agent_nums)))
                    for i in range(args.num_rounds - 1)
                ]
            spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
            temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((len(agent_names),len(agent_names)))
            if not graph.diff:
                loss_s = nuclear_norm(spatial_matrix_train)
                loss_t = nuclear_norm(temporal_matrix_train)
                frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
            else:
                loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
            sparsity_weight = graph.get_sparsity_weight(D)
            add_loss = sparsity_weight * (loss_s + loss_t) + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
            
            task = record["prompt"]
            test = record["test"]
            tests.append(test)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
            add_losses.append(add_loss)
            
        raw_results = await asyncio.gather(*answer_log_probs)

        raw_answers, log_probs, sp_per_round, tp_per_round, edge_counts = zip(*raw_results)
        
        batch_spatial = sum(sp for sp, _ in edge_counts)
        batch_temporal = sum(tp for _, tp in edge_counts)
        grand_spatial += batch_spatial
        grand_temporal += batch_temporal

        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        data = load_result(result_file)
               
        for task, answer, log_prob, add_loss, test in zip(current_batch, raw_answers, log_probs, add_losses, tests):
            if not isinstance(answer,list):
                raise TypeError(f"Expected a list for the answer, but got {type(answer).__name__}")
            answer = answer[0].lstrip("```python\n").rstrip("\n```")
            is_solved, _, _ = PyExecutor().execute(answer, [test], timeout=100)
            total_solved = total_solved + is_solved
            total_executed = total_executed + 1
            accuracy = total_solved/ total_executed
            utility = is_solved
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss+add_loss)
            updated_item = {
                "Question": task,
                "Tests": test,
                "Attempt answer": answer,
                "Solved": is_solved,
                "Solution": answer,
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy
            }
            data.append(updated_item)
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump(data, file, indent=4)
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Accuracy: {accuracy}")
        print("utilities:", utilities)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        print(f"spatial={grand_spatial}, temporal={grand_temporal}")


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        # random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix

    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
        # node_kwargs = [{'role':'Project Manager'}]

    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())
