# 部署简介
- 本 repo 是 LLM 部署在 Intel(R) GPU/NPU 平台上的实践，基于OpenVINO™ 工具套件
- OpenVINO™ 工具套件是一款开源工具套件，可以缩短延迟，提高吞吐量，加速 AI 推理过程，同时保持精度，缩小模型占用空间，优化硬件使用。它简化了计算机视觉、大型语言模型 (LLM) 和生成式 AI 等领域的 AI 开发和深度学习集成。
- Documents: https://docs.openvino.ai/2024/index.html

## 设备环境
- CPU: Intel(R) Core(TM) Ultra 7 258V   2.20 GHz
- GPU: Intel(R) Arc(TM) 140V GPU (16GB)
- RAM: 32GB LPDDR5X 8533MT/s

- Toolkit: OpenVINO

## Instruction
1. **安装 OpenVINO套件**
- 官方链接： https://docs.openvino.ai/2024/get-started/install-openvino.html?PACKAGE=OPENVINO_BASE&VERSION=v_2024_6_0&OP_SYSTEM=WINDOWS&DISTRIBUTION=PIP
```sh

# Step 1: Create virtual environment
python -m venv openvino_env
# Step 2: Activate virtual environment
openvino_env\Scripts\activate
# Step 3: Upgrade pip to latest version
python -m pip install --upgrade pip
# Step 4: Download and install the package
pip install openvino==2024.6.0
# Step 5 (Recommanded): Install openvino dev
pip install openvino-dev
```

2. **模型权重下载与转换**
- OpenVINO 支持 ONNX 格式的模型。为了进一步提高推理效率，这里先将 Hugging Face 下载下来的模型转换为 ONNX 格式，再转换为 OpenVINO 的 Intermediate Representation (IR) 格式。
- 关于 IR 格式的官方介绍：https://docs.openvino.ai/2023.3/openvino_ir.html#:~:text=OpenVINO%20IR%2C%20known%20as%20Intermediate%20Representation%2C%20is%20the,two%20files%3A%20an%20XML%20and%20a%20binary%20file.
- IR 转换操作将经常使用的深度学习操作转换为OpenVINO中各自的类似表示，并使用来自训练模型的相关权重和偏差对其进行调整。生成的IR包含两个文件：
    - xml - 描述模型拓扑
    - bin - 包含权重和二进制数据

```sh
# 执行该命令以进行模型转换
# 该命令还会使用 OpenVINO 的 benchmark_app 工具测试模型性能
python bulid_openvino_ir.py \
    --model_id Qwen/Qwen2.5-0.5B \
    --seq_len 128 \
    --device GPU
```
- 如果系统中有多个 GPU，可以通过 ```device_name="GPU.1"``` 指定具体的 GPU
- 可以在 Model Optimizer 中使用 ```--data_type FP16``` 将模型转换为 FP16 精度，以提高性能。相关代码位于 ```build_openvino_ir.py``` 中
- 亦可使用 OpenVINO 的异步推理来提高吞吐量：https://docs.openvino.ai/2024/notebooks/async-api-with-output.html
- 在以上参数的设定下，单张 Intel(R) Arc(TM) 140V GPU 的 benchmark 结果如下：
```sh
[Step 11/11] Dumping statistics report
[ INFO ] Execution Devices:['GPU.0']
[ INFO ] Count:            2360 iterations
[ INFO ] Duration:         60150.14 ms
[ INFO ] Latency:
[ INFO ]    Median:        101.68 ms
[ INFO ]    Average:       101.73 ms
[ INFO ]    Min:           45.16 ms
[ INFO ]    Max:           177.16 ms
[ INFO ] Throughput:   39.24 FPS
```
- 注：截至2025.02.01，本 repo 所使用设备的 NPU 尚未支持某些算子，例如当设置 ```--device NPU``` 时，你会收到如下错误（NPU 不支持 Range 算子（opset4 版本）：
```
ZE_RESULT_ERROR_INVALID_ARGUMENT, code 0x78000004 - generic error code for invalid arguments . 
[NOT IMPLEMENTED] Unsupported operation /model/Range with type Range and version opset4. Try to update the driver to the latest version. If the error persists, please submit a bug report in https://github.com/openvinotoolkit/openvino/issues
```


3. **构建简单的API，部署到生产环境**
- 安装环境依赖
```
pip install openvino fastapi uvicorn
```
- 将 OpenVINO 模型封装为一个 Python 类，方便调用和管理。代码位于 ```openvino_model.py``` 中
- 使用 FastAPI 将模型封装为 RESTful API。代码位于 ```api_server.py``` 中
- 构建 POST 请求，调用 API 服务。代码位于 ```api_client.py``` 中

```sh
# 1 在一个终端中，启动 API 服务
python api_server.py
# 2 启动另一个终端，调用 API 服务
python api_client.py

# 若api_server服务启动成功，终端会输出如下结果
# INFO:     Started server process [5660]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

4. **进一步封装并发布应用**
- 可使用 PyInstaller 打包为可执行文件
- 可使用 Docker 容器化部署

## Reference
- http://cloxtec.com/?thread-285.htm
- https://docs.openvino.ai/2024/documentation.html
