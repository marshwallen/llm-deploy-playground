# Reference: 
# http://cloxtec.com/?thread-285.htm
# https://docs.openvino.ai/2024/documentation.html

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import shutil
import subprocess
import argparse

def create_ir(args):
    """
    从 Hugging Face Hub 下载 LLM 模型，将其转换为 Pytorch onnx 格式，再通过 OpenVINO 转换为专有 IR 格式
    model_id: Hugging Face Hub 模型 ID
    """
    # 下载模型
    model_dir = os.path.join(os.getcwd(), "model")

    snapshot_download(args.model_id, cache_dir=model_dir)
    _lock_dir = os.path.join("model", ".locks")
    if os.path.exists(_lock_dir):
        shutil.rmtree(_lock_dir)

    # 加载 Qwen 模型和对应的 tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        local_files_only=True,
        cache_dir=model_dir,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=True,
        cache_dir=model_dir,
        )

    # 创建示例输入（tokenized 输入）
    dummy_input = tokenizer("Hello, how are you?", return_tensors="pt")["input_ids"]

    # 导出为 ONNX 格式
    onnx_path = f"onnx/{args.model_id.split('/')[-1]}.onnx"
    os.makedirs("onnx", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=14,  # ONNX 版本
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},  # 支持动态 batch size 和 sequence length
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
    )
    print(f"Model exported to {onnx_path}")

    # Convert to onnx
    command = ["mo", "--input_model", f"onnx/{args.model_id.split('/')[-1]}.onnx", "--output_dir", "output/"]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Command output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e.stderr)

def benchmark(args):
    """
    使用 OpenVINO 的 benchmark_app 工具测试模型性能
    """
    command = ["benchmark_app", 
               "-m", f"output/{args.model_id.split('/')[-1]}.xml", 
               "-d", args.device, 
               "-shape", f"input_ids[1,{args.seq_len}]"
               ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Command output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Standard error:", e.stderr)  # 检查标准错误

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="Hugging Face Hub 模型 ID")
    parser.add_argument("--seq_len", type=int, default=128, help="用于 benchmark 的输入序列长度")
    parser.add_argument("--device", type=str, default="GPU", help="用于 benchmark 的推理设备，可选 CPU/GPU/NPU")
    args = parser.parse_args()

    if not os.path.exists(f"output/{args.model_id.split('/')[-1]}.xml"):
        create_ir(args)
    benchmark(args)