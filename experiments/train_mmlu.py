import torch
import torch.nn.functional as F
from typing import Iterator
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy
import random

from AgentDropout.graph.graph import Graph
from experiments.accuracy import Accuracy
from AgentDropout.utils.globals import Cost, PromptTokens, CompletionTokens
from AgentDropout.utils.utils import nuclear_norm,frobenius_norm
from ddc import calculate_D

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            imp_per_iters: int = 1,
            pruning_rate: float = 0.05,
            args=None,
            kwargs=None,
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
        while True:
            for idx in range(len(dataset)):
                record = dataset[idx]
                yield record
    
    loader = infinite_data_loader()
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
        
        if not graph.diff:
            optimizer = torch.optim.Adam(
                [graph.spatial_logits_1, graph.temporal_logits_1] + controller_params,
                lr=lr
            )
        else:
            optimizer = torch.optim.Adam(
                list(graph.spatial_logits_1.parameters()) +
                list(graph.temporal_logits_1.parameters()) +
                controller_params,
                lr=lr
            )      
        for i_iter in range(2):
            print(f"Train {i_iter}", 80*'-')
            start_ts = time.time()
            correct_answers = []
            answer_log_probs = []
            add_losses = []
            for i_record, record in zip(range(20), loader):
                sample_input_dict = dataset.record_to_input(record)
                sample_task = sample_input_dict["task"]
                D = calculate_D(sample_task)

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

                spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
                temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
                if not graph.diff:
                    loss_s = nuclear_norm(spatial_matrix_train)
                    loss_t = nuclear_norm(temporal_matrix_train)
                    frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                    frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                    loss_cs = connectivity_loss(spatial_matrix_train)
                    loss_ct = connectivity_loss(temporal_matrix_train)
                else:
                    loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                    loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                    loss_cs = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in spatial_matrix_train]))
                    loss_ct = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in temporal_matrix_train]))
                    frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                    frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
                sparsity_weight = graph.get_sparsity_weight(D)
                add_loss = sparsity_weight * (loss_s + loss_t) + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
                add_loss = 0
                input_dict = dataset.record_to_input(record)
                if args.dec:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds,skip=True)))
                else:
                    answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
                correct_answer = dataset.record_to_target_answer(record)
                correct_answers.append(correct_answer)
                add_losses.append(add_loss)
                
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs, sp_per_round, tp_per_round, edge_counts = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            answers: List[str] = []
            
            for raw_answer, log_prob, add_loss, correct_answer in zip(raw_answers, log_probs, add_losses, correct_answers):
                answer = dataset.postprocess_answer(raw_answer)
                answers.append(answer)
                assert isinstance(correct_answer, str), \
                        f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                accuracy = Accuracy()
                accuracy.update(answer, correct_answer)
                utility = accuracy.get()
                utilities.append(utility)
                single_loss = - log_prob * utility
                loss_list.append(single_loss+add_loss)
                print(f"correct answer:{correct_answer}")
        
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
            
            print("raw_answers:",raw_answers)
            print("answers:",answers)
            print(f"Batch time {time.time() - start_ts:.3f}")
            print("utilities:", utilities)
            print("loss:", total_loss.item())
            print("Spatial logits Grad:", graph.spatial_logits_1[0].grad)
            print("Spatial logits:", graph.spatial_logits_1)
            print("Temporal logits:", graph.temporal_logits_1)
            print("Spatial probs:", spatial_probs)
            print("Temporal probs:", temporal_probs)
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")
        if isinstance(graph.spatial_masks, torch.nn.ParameterList):
            old_spatial = [mask.detach().clone() for mask in graph.spatial_masks]
            old_temporal = [mask.detach().clone() for mask in graph.temporal_masks]
        else:
            old_spatial = graph.spatial_masks.detach().clone()
            old_temporal = graph.temporal_masks.detach().clone()

        # 2. 调用 DEC 阶段剪枝
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
    loader = infinite_data_loader()

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
    graph.optimized_spatial=True
    graph.optimized_temporal=True

    for i_iter in range(2):
        print(f"Train {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []
        add_losses = []
        for i_record, record in zip(range(10), loader):
            sample_input_dict = dataset.record_to_input(record)
            sample_task = sample_input_dict["task"]
            D = calculate_D(sample_task)
            graph.adjust_graph_structure(D)
            graph.set_difficulty_tensor(torch.tensor(D, dtype=torch.float32))
            realized_graph = copy.deepcopy(graph)

            #realized_graph.spatial_logits = graph.spatial_logits
            #realized_graph.temporal_logits = graph.temporal_logits
            
            if not graph.diff:
                spatial_matrix_train = realized_graph.spatial_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))
                temporal_matrix_train = realized_graph.temporal_logits.reshape((sum(args.agent_nums),sum(args.agent_nums)))

            else:
                spatial_matrix_train = [
                        (realized_graph.base_spatial_logits[i] + realized_graph.spatial_logit_mlps[i](graph.current_difficulty_tensor)).reshape((sum(args.agent_nums), sum(args.agent_nums)))
                        for i in range(args.num_rounds)
                    ]
                temporal_matrix_train = [
                        (realized_graph.base_temporal_logits[i] + realized_graph.temporal_logit_mlps[i](graph.current_difficulty_tensor)).reshape((sum(args.agent_nums), sum(args.agent_nums)))
                        for i in range(args.num_rounds - 1)
                    ]

            spatial_matrix_fixed = torch.tensor(kwargs["fixed_spatial_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
            temporal_matrix_fixed = torch.tensor(kwargs["fixed_temporal_masks"],dtype=torch.float32).reshape((sum(args.agent_nums),sum(args.agent_nums)))
            # spatial_matrix_fixed = spatial_matrix_fixed[:4,:4]
            # temporal_matrix_fixed = temporal_matrix_fixed[:4,:4]
            if not graph.diff:
                loss_s = nuclear_norm(spatial_matrix_train)
                loss_t = nuclear_norm(temporal_matrix_train)
                frob_loss_s = frobenius_norm(spatial_matrix_fixed, spatial_matrix_train)
                frob_loss_t = frobenius_norm(temporal_matrix_fixed, temporal_matrix_train)
                loss_cs = connectivity_loss(spatial_matrix_train)
                loss_ct = connectivity_loss(temporal_matrix_train)
                # print(loss_cs)
            else:
                # loss_s = sum(nuclear_norm(matrix) for matrix in spatial_matrix_train)
                # loss_t = sum(nuclear_norm(matrix) for matrix in temporal_matrix_train)
                # frob_loss_s = sum(frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train)
                # frob_loss_t = sum(frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train)
                loss_s = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in spatial_matrix_train]))
                loss_t = torch.mean(torch.stack([nuclear_norm(matrix) for matrix in temporal_matrix_train]))
                loss_cs = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in spatial_matrix_train]))
                loss_ct = torch.mean(torch.stack([connectivity_loss(matrix) for matrix in temporal_matrix_train]))
                frob_loss_s = torch.mean(torch.stack([frobenius_norm(spatial_matrix_fixed, matrix) for matrix in spatial_matrix_train]))
                frob_loss_t = torch.mean(torch.stack([frobenius_norm(temporal_matrix_fixed, matrix) for matrix in temporal_matrix_train]))
            sparsity_weight = graph.get_sparsity_weight(D)
            add_loss = sparsity_weight * (loss_s + loss_t) + F.relu(frob_loss_s - args.delta) + F.relu(frob_loss_t - args.delta)
            # if graph.diff:
            # add_loss = 0
            input_dict = dataset.record_to_input(record)
            print(input_dict)
            if args.dec:
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            else:
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
            add_losses.append(add_loss)
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs, sp_per_round, tp_per_round, edge_counts = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        
        for raw_answer, log_prob, add_loss, correct_answer in zip(raw_answers, log_probs, add_losses, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss+add_loss)
            print(f"correct answer:{correct_answer}")
    
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
        
        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)
        print("loss:", total_loss.item()) 
        # print("Spatial logits Grad:", graph.spatial_logits.grad)
        # print("Temporal logits Grad:", graph.spatial_logits.grad)
        for i in range(args.num_rounds):
                    logits_i = graph.base_spatial_logits[i] + graph.spatial_logit_mlps[i](graph.current_difficulty_tensor)
                    print(f"Spatial logits (round {i}):", logits_i)
        for i in range(args.num_rounds - 1):
                    logits_i = graph.base_temporal_logits[i] + graph.temporal_logit_mlps[i](graph.current_difficulty_tensor)
                    print(f"Temporal logits (round {i}):", logits_i)
        # if graph.dec_1:
        #     print("Decision logits:", graph.decision_logits)
        print("Spatial probs:", spatial_probs)
        print("Temporal probs:", temporal_probs)
        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")
        
        if (i_iter+1)%2 == 0:
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
    return saved_spatial_masks, saved_temporal_masks
    # graph.optimized_spatial=False
    # graph.optimized_temporal=False        


def connectivity_loss(A: torch.Tensor) -> torch.Tensor:

    expA = torch.matrix_exp(A)

    penalty = torch.trace(expA) - A.shape[0]
    return penalty

def set_seed(seed):
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def remove_zero_rows_and_columns(matrix,row_to_delete,col_to_delete):
    new_matrix = torch.cat([matrix[:row_to_delete], matrix[row_to_delete+1:]])
    new_matrix = torch.cat([new_matrix[:, :col_to_delete], new_matrix[:, col_to_delete+1:]], dim=1)

    return new_matrix