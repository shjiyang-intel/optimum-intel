# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional, Tuple, Union
import inspect
from pathlib import Path

import torch
from transformers import PreTrainedModel, GenerationConfig, GenerationMixin, AutoConfig
from transformers.generation.utils import GenerateOutput
from transformers.utils import logging
import os

try:
    import openvino_genai
    openvino_genai_available = True
except ImportError:
    openvino_genai_available = False

from openvino_genai.py_openvino_genai import TokenizedInputs
import openvino as ov
from openvino import Tensor

logger = logging.get_logger(__name__)


class OpenVINOGenAIModel:
    """
    Base class for OpenVINO GenAI models.
    """

    def __init__(self, model_path: Union[str, Path], **kwargs):
        if not openvino_genai_available:
            raise ImportError("OpenVINO GenAI is not available. Please install it with `pip install openvino-genai`")
            
        self.model_path = str(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        
        self.device = kwargs.get("device", "cpu")
        if isinstance(self.device, str):
            self.device = self.device.upper()  

        self.device_is_npu = self.device == "NPU"

        self._setup_ov_genai(**kwargs)

    @property
    def ov_tokenizer(self):
        return self._ov_tokenizer

    def _setup_ov_genai(self, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Forward method for the OpenVINO GenAI model.
        Base implementation for non-generative tasks.
        """
        raise NotImplementedError(
            "Direct forward pass is not implemented for OpenVINOGenAIModel. "
            "Please use the specific task model or the generate method for text generation."
        )


class OpenVINOGenAIModelForCausalLM(OpenVINOGenAIModel, GenerationMixin):
    """
    OpenVINO GenAI model for causal language modeling.
    """

    def __init__(self, 
        model_path: Union[str, Path], 
        **kwargs):
        super().__init__(model_path, **kwargs)
        
        self.generation_config = None

    def __call__(self, 
                 inputs: TokenizedInputs, 
                 generation_config: Optional[openvino_genai.GenerationConfig] = None, 
                 **kwargs):
        return self.forward(inputs, generation_config, **kwargs)

    def _setup_ov_genai(self, **kwargs):
        """
        Setup the OpenVINO GenAI environment for the model.
        """
        logger.info("Setting up OpenVINO GenAI for causal language model")
        # Extract config dictionary from kwargs if any
        config_dict = kwargs.pop("config", {})        
        if not isinstance(config_dict, dict):
            config_dict = {}
        
        # Since we already have self.device set at base class, ensure it's not passed twice
        kwargs.pop("device", None)
        if self.device != "NPU":
            if "MAX_PROMPT_LEN" in kwargs.keys():
                kwargs.pop("MAX_PROMPT_LEN", None)                 
            if "MIN_RESPONSE_LEN" in kwargs.keys():
                kwargs.pop("MIN_RESPONSE_LEN", None)
        else:
            self._max_prompt_len = kwargs.get("MAX_PROMPT_LEN", None)

        # Remove all kwargs with None values
        none_keys = [k for k, v in kwargs.items() if v is None]
        for k in none_keys:
            kwargs.pop(k)
        
        self.ov_genai_pipeline = openvino_genai.LLMPipeline(
            models_path=self.model_path,
            device=self.device,
        )

        tokenizer_path = os.path.join(self.model_path, "openvino_tokenizer.xml")
        if not os.path.exists(tokenizer_path):
            logger.info("OpenVINO tokenizer not found. Converting from HF tokenizer...")
            try:
                from transformers import AutoProcessor
                from openvino_tokenizer import convert_tokenizer
            except ImportError:
                raise ImportError(
                    "OpenVINO tokenizer not found and openvino_tokenizer package is not available. "
                    "Please install it with `pip install openvino-tokenizer`"
                )
                
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            processor.tokenizer.save_pretrained(self.model_path)
            convert_tokenizer(
                processor.tokenizer, 
                with_detokenizer=True
            )
    
        self._ov_tokenizer = self.ov_genai_pipeline.get_tokenizer()

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        Prepare inputs for generation.
        Required for GenerationMixin compatibility.
        """
        # Implementing a minimal version for compatibility with GenerationMixin
        inputs = {"input_ids": input_ids}
        if "attention_mask" in kwargs:
            inputs["attention_mask"] = kwargs["attention_mask"]
        return inputs

    def forward(self, 
                inputs, 
                generation_config: Optional[openvino_genai.GenerationConfig] = None, 
                **kwargs):
        if isinstance(inputs, TokenizedInputs) or isinstance(inputs, Tensor):
            inputs = inputs
        else:
            if self.device_is_npu:
                inputs = self._ov_tokenizer.encode(inputs, max_length=self._max_prompt_len, pad_to_max_length=True)
            else:
                inputs = self._ov_tokenizer.encode(inputs)

        # self.logging_object_attributes(generation_config)

        res = self.ov_genai_pipeline.generate(
            inputs=inputs,
            generation_config=generation_config,
            **kwargs
        )
        # if isinstance(res, openvino_genai.EncodedResults):
        #     # res.token is batched sequence, here we only process 1st batch
        #     print('output string:', self._ov_tokenizer.decode(res.tokens[0]))
        # FIXME_SHJI: log_probs instead of logits
        return res.tokens[0], res.scores[0], res.logits[0]

    def _convert_transformers_config_to_openvino_genai(self, transformers_config, **kwargs):
        """
        Convert a transformers GenerationConfig to an openvino_genai GenerationConfig.
        Additional kwargs override any values from the config.
        """        
        ov_config = openvino_genai.GenerationConfig()
        
        if transformers_config is not None:
            config_dict = transformers_config.to_dict()
            
            for key, value in config_dict.items():
                if hasattr(ov_config, key):
                    setattr(ov_config, key, value)

        for key, value in kwargs.items():
            if hasattr(ov_config, key):
                setattr(ov_config, key, value)
        
        return ov_config

    def generate(
        self,
        inputs=None,
        generation_config=None,
        logits_processor=None,
        stopping_criteria=None,
        prefix_allowed_tokens_fn=None,
        synced_gpus=False,
        **kwargs
    ):
        if self.generation_config is None:
            self.generation_config = self._convert_transformers_config_to_openvino_genai(
                generation_config, **kwargs
            )
            self.logging_object_attributes(self.generation_config)
        
        if self.device_is_npu:
            inputs = self._ov_tokenizer.encode(inputs, max_length=self._max_prompt_len, pad_to_max_length=True)

        res = self.ov_genai_pipeline.generate(
            inputs=inputs,
            generation_config=self.generation_config,
            **kwargs
        )

        if isinstance(res, openvino_genai.EncodedResults):
            # res.token is batched sequence, here we only process 1st batch
            return self._ov_tokenizer.decode(res.tokens[0])
        
        return res
    
    """
    Debugging
    """
    def logging_object_attributes(self, obj):
        """Print all non-private attributes of an object."""
        print(f"{type(obj).__name__} Attributes:")
        
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                value = getattr(obj, attr_name)
                if not callable(value):
                    print(f"  {attr_name}: {value}")
