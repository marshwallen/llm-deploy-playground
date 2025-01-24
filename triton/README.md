## 部署简介
本 repo 采用 Triton 框架 + vLLM 推理后端的方式完成 LLM 的部署，可通过 API 访问
- vLLM: https://github.com/vllm-project/vllm
- Triton: https://github.com/triton-inference-server/server

## Triton 推理服务的整体架构
- 单节点推理服务
- 支持异构，比如：CPU、GPU、TPU等
- 支持多框架，比如：TensorRT、Pytorch、FasterTransformer等

![Triton](https://pica.zhimg.com/v2-625bf16c17f968303deeecdccd292134_1440w.jpg)

## Instruction
1. **安装环境依赖**
```sh
# pip 环境
pip install -r requirements.txt
# 拉取triton镜像
sudo docker pull nvcr.io/nvidia/tritonserver:24.12-py3
```

2. **在 Triton 容器中部署 vLLM**
- 这里先拉取 triton docker 容器，进入该容器安装 vLLM 后重新 commit
```sh
# 这一步的前提是安装好 nvidia-container-toolkit
sudo docker run --gpus all -it --rm nvcr.io/nvidia/tritonserver:24.12-py3 /bin/bash
sudo apt install vllm -i [可选镜像源]

# 另外打开一个终端，commit镜像
sudo docker commit [container_id] tritonserver:24.12-vllm-py3-custom
```

3. **准备预训练模型**
- 本 repo 以 RTX 3060 Ti 8G 单卡，Qwen2.5-0.5B为例子
```sh
# 从modelscope下载模型，并自动构建模型目录
python prepare_qwen_pt.py
```

4. **启动 Triton 服务**
```sh
sudo sh start_triton.sh
# 注：模型和缓存在显存占用中的比例可在模型权重目录下的 model.json 中配置（gpu_memory_utilization）
# 这里会开放三个端口，其中8000对应HTTP请求，8001对应GRPC请求

# 启动成功后可见其服务状态
# I0121 06:41:55.135739 1 grpc_server.cc:2558] "Started GRPCInferenceService at 0.0.0.0:8001"
# I0121 06:41:55.135989 1 http_server.cc:4725] "Started HTTPService at 0.0.0.0:8000"
# I0121 06:41:55.182729 1 http_server.cc:358] "Started Metrics Service at 0.0.0.0:8002"
```

5. **验证 Triton 服务是否正常工作，能否正常返回数据**
```sh
# 检查 docker 服务状态
sudo docker ps
# CONTAINER ID   IMAGE                                COMMAND                  CREATED              STATUS              PORTS                                                                                                                                         NAMES
# a4849af71c1c   tritonserver:24.12-vllm-py3-custom   "/opt/nvidia/nvidia_…"   About a minute ago   Up About a minute   0.0.0.0:18999->8000/tcp, [::]:18999->8000/tcp, 0.0.0.0:18998->8001/tcp, [::]:18998->8001/tcp, 0.0.0.0:18997->8002/tcp, [::]:18997->8002/tcp   quizzical_wescoff

# Http 客户端
python client_http.py

# grpc 客户端
python client_grpc.py
```

6. **后端服务实现**
- ```model_template.py``` 定义了模型如何与vLLM交互并提供服务。具体实现可在该文件修改。

## Reference
- [模型推理服务工具综述](https://zhuanlan.zhihu.com/p/721395381)
- [模型推理服务化框架Triton保姆式教程（一）：快速入门](https://zhuanlan.zhihu.com/p/629336492)
- [模型推理服务化框架Triton保姆式教程（二）：架构解析](https://zhuanlan.zhihu.com/p/634143650)
- [模型推理服务化框架Triton保姆式教程（三）：开发实践](https://zhuanlan.zhihu.com/p/634444666)