from typing import Dict, List, Tuple, Callable
import os

from experiments.instance import Instance


class ResultEntry:
    
    def __init__(self) -> None:
        self.is_success: bool = False
        self.runtime: float = float("inf")
        self.cost: float = float("inf")
    
    @property
    def key_names(self) -> List[str]:
        return ["is_success", "runtime", "cost"]
    
    def to_dict(self) -> Dict:
        return {
            "is_success": self.is_success,
            "runtime": self.runtime,
            "cost": self.cost
        }


class MPResultEntry(ResultEntry):
    
    def __init__(self, success:bool, runtime:float, cost:float, n_expanded:int=0, n_generated:int=0) -> None:
        self.is_success = success
        self.runtime = runtime
        self.cost = cost
        self.num_expanded_nodes = n_expanded
        self.num_generated_nodes = n_generated

    @property
    def key_names(self) -> List[str]:
        return super().key_names + ["num_expanded_nodes", "num_generated_nodes"]

    def to_dict(self) -> dict:
        base_dict = super().to_dict()
        base_dict.update({
            "num_expanded_nodes": self.num_expanded_nodes,
            "num_generated_nodes": self.num_generated_nodes
        })
        return base_dict


class Planner:
    
    @property
    def name(self) -> str:
        raise NotImplementedError("Planner.name() must be implemented in subclasses.")
    
    def run(self, istc:Instance, query) -> ResultEntry:
        raise NotImplementedError("Planner.run() must be implemented in subclasses.")

