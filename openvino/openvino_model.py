# 搭建 OpenVINOModel 类，处理模型推理
from openvino.runtime import Core
from transformers import AutoTokenizer
import os
import numpy as np

class OpenVINOModel:
    def __init__(self, model_id, device="GPU"):
        # 初始化 OpenVINO Runtime
        ie = Core()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=True,
            cache_dir=os.path.join(os.getcwd(), "model"),
            )
        
        # 加载模型
        model_path = f"output/{model_id.split('/')[-1]}.xml"
        self.compiled_model = ie.compile_model(model=model_path, device_name=device)  # 使用 GPU（Intel Arc）

        # 获取输入和输出层
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, message):
        """
        推理函数
        """
        # 准备输入数据（将文本转换为模型输入格式）
        instruct = f"{message[0]['role']}\n{message[0]['content']}\n{message[1]['role']}\n{message[1]['content']}\n"
        inputs = self.tokenizer(instruct, return_tensors="np", padding=True, truncation=True)
        input_data = inputs["input_ids"].astype(np.int64)  # 转换为 int64 类型
        results = self.compiled_model([input_data])[self.output_layer]

        # 后处理：将 logits 转换为 token IDs
        token_ids = np.argmax(results, axis=-1)  # 取 logits 中概率最大的 token ID
        output_text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)

        return output_text