## Triton的定位
- 单节点推理服务
- 支持异构，比如：CPU、GPU、TPU等
- 支持多框架，比如：TensorRT、Pytorch、FasterTransformer等

## Triton 推理服务的整体架构
![Triton](https://pica.zhimg.com/v2-625bf16c17f968303deeecdccd292134_1440w.jpg)

## Triton 仓库
https://github.com/triton-inference-server/server

## Instruction
- 安装环境依赖
```sh
# pip 环境
pip install -r requirements.txt

# 拉取triton镜像
sudo docker pull nvcr.io/nvidia/tritonserver:24.12-py3

# 进入该 docker 容器，安装vLLM后commit
# 这一步的前提是安装好 nvidia-container-toolkit
sudo docker run --gpus all -it --rm nvcr.io/nvidia/tritonserver:24.12-py3 /bin/bash
sudo apt install vllm -i [可选镜像源]

# 另外打开一个终端，commit镜像
sudo docker commit [container_id] tritonserver:24.12-vllm-py3-custom
```

- 准备预训练模型（本项目以8G显存单卡，Qwen2.5-0.5B为例子）
```sh
# 从modelscope下载模型，并自动构建模型目录
python prepare_qwen_pt.py
```

- 启动 Triton 服务
```sh
sudo sh start_triton.sh
# 注：模型和缓存在显存占用中的比例可在模型权重目录下的 model.json 中配置（gpu_memory_utilization）
# 这里会开放三个端口，其中8000对应HTTP请求，8001对应GRPC请求
```

- 验证 Triton 服务是否正常工作，能否正常返回数据
```sh
# Http 客户端
python client_http.py

# grpc 客户端
python client_grpc.py
```

## Reference
- [模型推理服务工具综述](https://zhuanlan.zhihu.com/p/721395381)
- [模型推理服务化框架Triton保姆式教程（一）：快速入门](https://zhuanlan.zhihu.com/p/629336492)
- [模型推理服务化框架Triton保姆式教程（二）：架构解析](https://zhuanlan.zhihu.com/p/634143650)
- [模型推理服务化框架Triton保姆式教程（三）：开发实践](https://zhuanlan.zhihu.com/p/634444666)