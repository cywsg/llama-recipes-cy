# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass, field
from typing import ClassVar, List

@dataclass
class lora_config:
     r: int=16
     lora_alpha: int=64
     target_modules: List[str]=field(default_factory=lambda: ["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"])
     bias= "none"
     task_type: str="CAUSAL_LM"
     lora_dropout: float=0.1
     inference_mode: bool=False
     # target_modules: ClassVar[List[str]] = ["k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj"]

@dataclass
class llama_adapter_config:
     adapter_len: int=10
     adapter_layers: int=30
     task_type: str="CAUSAL_LM"

@dataclass
class prefix_config:
     num_virtual_tokens: int=30
     task_type: str="CAUSAL_LM"    
