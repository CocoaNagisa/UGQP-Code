from __future__ import annotations

from dataclasses import dataclass


                                
                                                
ACTIONS = ["local", "edge_s", "edge_m", "edge_l", "cloud"]
ACTION_TO_ID = {name: idx for idx, name in enumerate(ACTIONS)}
ID_TO_ACTION = {idx: name for name, idx in ACTION_TO_ID.items()}

                              
                                             
NODE_TYPES = ["intent", "extract", "retrieve", "tool_call", "plan", "generate", "verify", "summarize"]
NODE_TYPE_TO_ID = {name: idx for idx, name in enumerate(NODE_TYPES)}


@dataclass
class EnvConfig:
                                                    
    max_nodes: int = 12
                                          
    min_nodes: int = 5
                                                   
    edge_count: int = 3
                                   
    seed: int = 2026
                                                                           
    variant: str = "full"
                                             
    task_graph_mode: str = "standard"
                            
    arrival_rate: float = 0.7
                                                                          
                                                                
    queue_mode: str = "bursty"
    queue_scale: float = 1.0
    queue_burst_probability: float = 0.18
    queue_burst_multiplier: float = 3.5
    queue_burst_min_steps: int = 2
    queue_burst_max_steps: int = 5
                                                                           
                                                              
    profile_mode: str = "tradeoff_v2"
                                                                                
    permute_node_order: bool = False
                                   
    network_scale: float = 1.0
                                  
    deadline_scale: float = 1.0
                  
    reward_latency_weight: float = 1.00
                                                                      
    reward_latency_normalizer: float = 5.0
                  
    reward_cost_weight: float = 0.20
                  
    reward_quality_weight: float = 2.40
                                                                      
    quality_propagation_strength: float = 1.50
                                     
    reward_deadline_weight: float = 1.20
                  
    reward_balance_weight: float = 0.18
                                                
    invalid_action_penalty: float = 8.0


def make_config(**kwargs) -> EnvConfig:
                                  
    cfg = EnvConfig()
    for key, value in kwargs.items():
                                       
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown config key: {key}")
        setattr(cfg, key, value)
    return cfg
