# 准备 Qwen2-VL-2B 数据集权重
from modelscope import snapshot_download
import os
import shutil

model_name = "Qwen2.5-0.5B"
data_dir = f"model_repository/Qwen2.5-0.5B/1/"
os.makedirs(data_dir, exist_ok=True)

snapshot_download(f'Qwen/Qwen2.5-0.5B', 
                    cache_dir="cache", 
                    revision='master')
download_dir = os.listdir("cache/Qwen")[0]
os.rename(f'cache/Qwen/{download_dir}', f'cache/Qwen/Qwen2.5-0.5B')
shutil.move(f'cache/Qwen/Qwen2.5-0.5B', data_dir)
shutil.rmtree('cache/')

# triton 启动配置文件
pbtxt = """name: "Qwen2.5-0.5B"
backend: "python"
max_batch_size: 0
input [
            {name: "prompt", data_type: TYPE_STRING, dims: [1]},
            {name: "stream", data_type: TYPE_BOOL, dims: [1], optional: True},
            {name: "sampling_parameters", data_type: TYPE_STRING, dims: [1], optional: True}
        ]
output [
            {name: "response", data_type: TYPE_STRING, dims: [-1]}
        ]
model_transaction_policy { decoupled: True}
instance_group [
  {
      count: 1
      kind: KIND_GPU
      gpus: [ 0 ]
  }
]"""

with open(f"model_repository/Qwen2.5-0.5B/config.pbtxt", "w") as f:
    f.write(pbtxt)

# vLLM 启动配置文件
vllm_json = """{
    "model": "Qwen2.5-0.5B",
    "tokenizer": "Qwen2.5-0.5B",
    "trust_remote_code":"true",
    "disable_log_requests": "true",
    "gpu_memory_utilization": 0.8
}"""

with open(data_dir+"/model.json", "w") as f:
    f.write(vllm_json)

shutil.copyfile("model_template.py", data_dir+"/model.py")
