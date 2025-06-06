import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio

from DAGP.graph.node import Node
from DAGP.agents.agent_registry import AgentRegistry
import random

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                diff:bool = False,
                dec:bool = False,
                rounds: int = 1,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                ):
        self.controller_weights = torch.nn.Parameter(torch.tensor([1.0, 0.5, 0.1]))
        self.controller_bias = torch.nn.Parameter(torch.tensor([0.0]))
        self.dynamic_pruning_rate = 0.1

        self.sparsity_base = torch.nn.Parameter(torch.tensor(0.5))
        self.sparsity_delta1 = torch.nn.Parameter(torch.tensor(0.5))
        self.sparsity_delta2 = torch.nn.Parameter(torch.tensor(0.5))

        self.fixed_spatial_masks = torch.tensor(fixed_spatial_masks)
        self.fixed_temporal_masks = torch.tensor(fixed_temporal_masks)
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        # print(fixed_temporal_masks)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        self.diff=diff
        self.rounds=rounds
        # self.dec=False
        self.dec_1=False
        self.skip_nodes = []
        
        self.init_nodes() # add nodes to the self.nodes
        self.init_potential_edges() # add potential edges to the self.potential_spatial/temporal_edges
        
        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        print(self.potential_spatial_edges)
        if dec:
            self.decision_logits = torch.ones(5) * torch.log(torch.tensor(1.0))
        if not diff:
            self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks
            self.base_spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,requires_grad=optimized_spatial) # trainable edge logits
            self.base_temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,requires_grad=optimized_temporal) # trainable edge logits
            self.spatial_logit_mlp = torch.nn.Linear(3, len(self.potential_spatial_edges), bias=False)
            self.temporal_logit_mlp = torch.nn.Linear(3, len(self.potential_temporal_edges), bias=False)
            self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
            if dec:
                self.spatial_logits_1 = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,requires_grad=optimized_spatial)
                self.temporal_logits_1 = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,requires_grad=optimized_temporal)
        else:
            if dec:
                self.spatial_logits_1 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,requires_grad=optimized_spatial) for _ in range(rounds)])
                self.temporal_logits_1 = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,requires_grad=optimized_temporal) for _ in range(rounds-1)])
            self.spatial_masks = torch.nn.ParameterList([torch.nn.Parameter(fixed_spatial_masks.clone(), requires_grad=False) for _ in range(rounds)])  # fixed edge masks
            self.base_spatial_logits = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit)
                for _ in range(rounds)
            ])
            self.spatial_logit_mlps = torch.nn.ModuleList([
                torch.nn.Linear(3, len(self.potential_spatial_edges), bias=False) for _ in range(rounds)
            ])

            self.base_temporal_logits = torch.nn.ParameterList([
                torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit)
                for _ in range(rounds-1)
            ])
            self.temporal_logit_mlps = torch.nn.ModuleList([
                torch.nn.Linear(3, len(self.potential_temporal_edges), bias=False) for _ in range(rounds-1)
            ])
            self.temporal_masks = torch.nn.ParameterList([torch.nn.Parameter(fixed_temporal_masks.clone(), requires_grad=False) for _ in range(rounds-1)])
    def get_sparsity_weight(self, D: tuple):
        """
        Calculate monotonic sparsity weight based on learnable parameters.
        """
        easy, medium, hard = D
        delta1 = torch.nn.functional.softplus(self.sparsity_delta1)
        delta2 = torch.nn.functional.softplus(self.sparsity_delta2)
        w2 = self.sparsity_base
        w1 = w2 + delta2
        w0 = w1 + delta1
        return easy * w0 + medium * w1 + hard * w2

    def adjust_graph_structure(self, D: tuple):
        """
        根据 (easy, medium, hard) 难度调整 self.dynamic_pruning_rate
        """
        D_tensor = torch.tensor(D, dtype=torch.float32).to(self.controller_weights.device)
        w = torch.softmax(self.controller_weights, dim=0)  # 权重归一化
        pruning_score = torch.dot(w, D_tensor) + self.controller_bias  # scalar
        pruning_score = torch.sigmoid(pruning_score)  # squash to [0, 1]
        self.dynamic_pruning_rate = torch.clamp(pruning_score, min=0.05, max=0.9).item()
        #self.dynamic_pruning_rate = torch.clamp(pruning_score, min=0.1, max=0.1).item()
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)



    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        adjusted_logits = self.base_spatial_logits

        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, adjusted_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            # print(out_node)
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                    # print(potential_connection)
                continue
            if not self.check_cycle(in_node, {out_node}):
                # print("@@@@@@@@@@@")
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    
    
    def construct_spatial_connection_diff(self, round:int = 0, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        adjusted_logits = self.base_spatial_logits[round]

        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, adjusted_logits, self.spatial_masks[round]):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            out_id = list(self.nodes).index(out_node.id)
            in_id = list(self.nodes).index(in_node.id)
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                # if round == self.rounds-1 and self.dec_1 and (self.decision_masks[in_id]==0 or self.decision_masks[out_id]==0):
                #     print(11111)
                #     continue
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                    # print(potential_connection)
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        adjusted_logits = self.base_temporal_logits + self.temporal_logit_mlp(self.current_difficulty_tensor)

        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, adjusted_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                    # print(potential_connection)
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    def set_difficulty_tensor(self, D_tensor: torch.Tensor):
        self.current_difficulty_tensor = D_tensor

    def construct_temporal_connection_diff(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        adjusted_logits = self.base_temporal_logits[round - 1] + self.temporal_logit_mlps[round - 1](self.current_difficulty_tensor)

        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, adjusted_logits, self.temporal_masks[round - 1]):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 6000,
                  skip: bool=False,
                  case: bool=False) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        log_probs_skip = 0
        all_answers = []
        spatial_round_counts = []
        temporal_round_counts = []
        for round in range(num_rounds):
            round_answers = {}
            if not self.diff:
                log_probs += self.construct_spatial_connection()
                spatial_round_counts.append(sum(len(n.spatial_successors) for n in self.nodes.values()))
                log_probs += self.construct_temporal_connection(round)
                temporal_round_counts.append(sum(len(n.temporal_successors) for n in self.nodes.values()))
            else:
                log_probs += self.construct_spatial_connection_diff(round)
                spatial_round_counts.append(sum(len(n.spatial_successors) for n in self.nodes.values()))
                log_probs += self.construct_temporal_connection_diff(round)
                temporal_round_counts.append(sum(len(n.temporal_successors) for n in self.nodes.values()))
            # print(self.num_edges)

            in_sp = {nid: len(node.spatial_predecessors)
                     for nid, node in self.nodes.items()}
            in_tm = {nid: len(node.temporal_predecessors)
                     for nid, node in self.nodes.items()}
            # 2) 只有两个都为 0 才入初始队列
            ready = [nid for nid in self.nodes
                     if in_sp[nid] == 0 and in_tm[nid] == 0]

            selected_index=-1

            if round <= 5 and skip:
                # log_probs = 0
                min_logit=100
                min_node=None
                min_loss=100
                loss_t_list = []
                loss_f_list = []
                log_list=[]
                for node_id, node in self.nodes.items():
                    in_id = list(self.nodes).index(node_id)
                    count=0
                    logits_count=0.
                    loss_t=0
                    loss_f=0
                    t=1.0
                    
                    for last_node in node.spatial_successors:
                        last_id = list(self.nodes).index(last_node.id)
                        count+=1
                        logits_count+=t*self.spatial_logits_1[round][in_id*5+last_id]
                        loss_t+=torch.log(1-torch.sigmoid(self.spatial_logits_1[round][in_id*5+last_id]))
                        loss_f+=torch.log(torch.sigmoid(self.spatial_logits_1[round][in_id*5+last_id]))
                    for last_node in node.spatial_predecessors:
                        last_id = list(self.nodes).index(last_node.id)
                        count+=1
                        # logits_count+=torch.sigmoid(t*self.spatial_logits_1[round][last_id*5+in_id])
                        logits_count+=t*self.spatial_logits_1[round][last_id*5+in_id]
                        loss_t+=torch.log(1-torch.sigmoid(self.spatial_logits_1[round][last_id*5+in_id]))
                        loss_f+=torch.log(torch.sigmoid(self.spatial_logits_1[round][last_id*5+in_id]))
                    log_list.append(logits_count)
                    if count==0:
                        count=1.0
                    loss_t_list.append(loss_t)
                    loss_f_list.append(loss_f)

                    
                p = torch.softmax(torch.tensor(log_list),dim=0)
                selected_index = torch.multinomial(p, num_samples=1, replacement=False)
                for i in range(5):
                    if i==selected_index:
                        log_probs_skip += 4.0*loss_t_list[i]
                    else:
                        log_probs_skip += loss_f_list[i]
            while ready:
                current = ready.pop(0)
                node = self.nodes[current]

                # 3) 如果节点既没有任何空间边也没有任何时序边，跳过不执行
                if (not node.spatial_predecessors
                        and not node.spatial_successors
                        and not node.temporal_predecessors
                        and not node.temporal_successors):
                    continue

                # 执行节点
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(
                            node.async_execute(input),
                            timeout=max_time
                        )
                        break
                    except Exception as e:
                        print(f"Error on node {current}: {e}")
                        tries += 1

                # 4) 执行完后，更新空间后继和时序后继的前驱数
                for succ in node.spatial_successors:
                    in_sp[succ.id] -= 1
                    if in_sp[succ.id] == 0 and in_tm[succ.id] == 0:
                        ready.append(succ.id)
                for succ in node.temporal_successors:
                    in_tm[succ.id] -= 1
                    if in_sp[succ.id] == 0 and in_tm[succ.id] == 0:
                        ready.append(succ.id)
            # —— 修改部分结束 —— #

            # 汇总本轮回答
            for nid, node in self.nodes.items():
                round_answers[node.role + nid] = node.outputs
            all_answers.append(round_answers)
            self.update_memory()
        


        # if self.dec_1==False:
        if len(self.potential_spatial_edges)>0:
            self.connect_decision_node()
            await self.decision_node.async_execute(input)
            final_answers = self.decision_node.outputs
        else:
            final_answers = list(self.nodes.values())[0].outputs

        total_spatial = sum(spatial_round_counts)
        total_temporal = sum(temporal_round_counts)

        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        if skip:
            return final_answers, log_probs_skip, spatial_round_counts, temporal_round_counts, (total_spatial, total_temporal)
        elif case:
            return final_answers, log_probs, all_answers, spatial_round_counts, temporal_round_counts, (total_spatial, total_temporal)
        else:
            return final_answers, log_probs, spatial_round_counts, temporal_round_counts, (total_spatial, total_temporal)

    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks
    def refine_masks_from_existing(self, pruning_rate: float, times: int = 2):
        if self.optimized_spatial:
            for i in range(self.rounds):
                for _ in range(times):  # apply multiple rounds
                    logits = self.base_spatial_logits[i]
                    logits = logits.detach().clone()
                    existing_mask = self.spatial_masks[i]
                    valid_indices = (existing_mask == 1).nonzero(as_tuple=False).flatten()
                    if len(valid_indices) <= 1:
                        continue
                    num_to_prune = max(1, int(pruning_rate * len(valid_indices)))
                    selected_logits = logits[valid_indices]
                    sorted_idx = torch.argsort(selected_logits)
                    prune_indices = valid_indices[sorted_idx[:num_to_prune]]
                    self.spatial_masks[i].data[prune_indices] = 0

        if self.optimized_temporal:
            for i in range(self.rounds - 1):
                for _ in range(times):
                    logits = self.base_temporal_logits[i] + self.temporal_logit_mlps[i](self.current_difficulty_tensor)
                    logits = logits.detach().clone()
                    existing_mask = self.temporal_masks[i]
                    valid_indices = (existing_mask == 1).nonzero(as_tuple=False).flatten()
                    if len(valid_indices) <= 1:
                        continue
                    num_to_prune = max(1, int(pruning_rate * len(valid_indices)))
                    selected_logits = logits[valid_indices]
                    sorted_idx = torch.argsort(selected_logits)
                    prune_indices = valid_indices[sorted_idx[:num_to_prune]]
                    self.temporal_masks[i].data[prune_indices] = 0
    
    def update_masks_diff(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            for i in range(self.rounds):
                num_edges = (self.spatial_masks[i] > 0).sum()
                num_masks = (self.spatial_masks[i] == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
                _edge_logits = self.base_spatial_logits[i]
                _edge_logits = _edge_logits.clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.spatial_masks[i] == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                # _edge_logits = torch.cat([_edge_logits[:num_masks], _edge_logits[num_masks:][torch.randperm(len(_edge_logits)-num_masks)]])
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.spatial_masks[i][prune_idx] = 0
        
        if self.optimized_temporal:
            for i in range(self.rounds-1):
                num_edges = (self.temporal_masks[i] > 0).sum()
                num_masks = (self.temporal_masks[i] == 0).sum()
                prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
                _edge_logits = self.base_temporal_logits[i] + self.temporal_logit_mlps[i](self.current_difficulty_tensor)
                _edge_logits = _edge_logits.clone()
                min_edge_logit = _edge_logits.min()
                _edge_logits[self.temporal_masks[i] == 0] = min_edge_logit - 1.0
                sorted_edges_idx = torch.argsort(_edge_logits)
                # _edge_logits = torch.cat([_edge_logits[:num_masks], _edge_logits[num_masks:][torch.randperm(len(_edge_logits)-num_masks)]])
                prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
                self.temporal_masks[i][prune_idx] = 0
        # self.spatial_logits_1 = self.spatial_logits.clone()
        # self.temporal_logits_1 = self.temporal_logits.clone()
        return self.spatial_masks, self.temporal_masks
    def get_dynamic_logits(self):
        if not self.diff:
            spatial_logits = self.base_spatial_logits
            temporal_logits = self.base_temporal_logits
        else:
            spatial_logits = [
                self.base_spatial_logits[i]
                for i in range(self.rounds)
            ]
            temporal_logits = [
                self.base_temporal_logits[i]
                for i in range(self.rounds - 1)
            ]
        return spatial_logits, temporal_logits
    def compute_masks_from_scratch(self, pruning_rate: float):
        if self.optimized_spatial:
            for i in range(self.rounds):
                edge_logits = self.base_spatial_logits[i]
                edge_logits = edge_logits.detach()
                total_edges = edge_logits.shape[0]
                num_agents = int(total_edges ** 0.5)

                # step 1: build initial mask with self-loop masked out
                mask = torch.ones_like(edge_logits)
                diag_indices = [a * num_agents + a for a in range(num_agents)]
                mask[diag_indices] = 0  # no self-loop

                # step 2: prune only on non-diagonal entries
                non_diag_indices = [i for i in range(total_edges) if i not in diag_indices]
                non_diag_logits = edge_logits[non_diag_indices]
                num_prune = max(1, int(pruning_rate * len(non_diag_indices)))
                sorted_local = torch.argsort(non_diag_logits)
                prune_global_idx = [non_diag_indices[j.item()] for j in sorted_local[:num_prune]]
                mask[prune_global_idx] = 0
                self.spatial_masks[i].data = mask
                #print("1____"+str(i))
                #print(mask)


        if self.optimized_temporal:
            for i in range(self.rounds - 1):
                edge_logits = self.base_temporal_logits[i] + self.temporal_logit_mlps[i](self.current_difficulty_tensor)
                edge_logits = edge_logits.detach()
                total_edges = edge_logits.shape[0]
                num_prune = max(1, int(pruning_rate * total_edges))
                sorted_idx = torch.argsort(edge_logits)
                mask = torch.ones_like(edge_logits)
                mask[sorted_idx[:num_prune]] = 0
                self.temporal_masks[i].data = mask
                #print("2____"+str(i))
                #print(mask)

    def update_masks_dec(self):
        spatial_matrix_train = [param.reshape((5, 5)) for param in self.spatial_logits_1]
        temporal_matrix_train = [param.reshape((5, 5)) for param in self.temporal_logits_1]

        for i in range(len(spatial_matrix_train)):
            min = 100
            min_node = -1
            for j in range(5):
                sum = torch.sum(spatial_matrix_train[i][j,:]).item() + torch.sum(spatial_matrix_train[i][:,j]).item()
                # if i >= 1:
                #     sum += torch.sum(temporal_matrix_train[i-1][j,:]).item() + torch.sum(temporal_matrix_train[i-1][:,j]).item()
                count = torch.sum(self.fixed_spatial_masks[j,:]).item() + torch.sum(self.fixed_spatial_masks[:,j]).item()
                sum = sum / count
                if sum < min:
                    min = sum
                    min_node = j
            # min_node=random.randint(0, 4)
            self.skip_nodes.append(min_node)
            for k in range(5):
                self.spatial_masks[i][min_node*5+k]=0
                self.spatial_masks[i][k*5+min_node]=0
            if i > 0:
                for k in range(5):
                    self.temporal_masks[i-1][k*5+min_node]=0
            if i < len(spatial_matrix_train) - 1:
                for k in range(5):
                    self.temporal_masks[i][min_node*5+k]=0
