# 构造 Langchain LLM 类，实现本地 LLM
# 这里以 Qwen2.5-3B-Instruct 为例

from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from modelscope import snapshot_download
import os

class LocalLangchainLLM(LLM):
    tokenizer : AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_name="Qwen/Qwen2.5-3B"):
        """
        基于本地模型自定义 LLM 类
        model_name: From https://huggingface.co/
        """
        # 从本地初始化模型
        super().__init__()

        # 如果不存在本地文件，下载
        model_dir = "model/" + model_name.replace(".","___")
        snapshot_download(model_name, cache_dir='./model', revision='master')

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, 
            device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_dir)
        self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.model = self.model.eval()

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        """
        重写调用函数
        """
        messages = [
            {"role": "system", "content": "你是一个人工智能助手"},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=64
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
        
    @property
    def _llm_type(self) -> str:
        return "LocalLangchainLLM"
    