# LLM-deploy-playground
- LLM 本地部署实践

## 😺 Introduction
- 推理服务部署参考：https://github.com/liguodongiot/llm-action

## 🤔 推理框架的基本要素
1. 得有一个部署平台/集群（如：Kubernetes）。用于对一个推理服务的生命周期的管理（模型的加载/卸载，模型服务实例的弹性扩缩容等）
2. 还应该有一个负载均衡器（如：Ingress），解决模型推理服务实例负载均衡的问题。将客户端的请求，均衡的分配给推理服务实例
3. 还有一个模型仓库（如：本地文件系统），对模型权重文件和模型输入输出配置进行管理
4. 之后还应该有一个模型监控服务，用于观察模型的宏观/微观的数据
5. 最后，也是最核心的，它还应该有一个推理服务，进行模型推理

## 😎 Triton Inference Server 部署
- NVIDIA 开源的商用级别的服务框架
- 跳转：https://github.com/marshwallen/llm-deploy-playground/tree/main/triton

## 😎 LangChain 部署
- LangChain 是一个用于开发由语言模型驱动的应用程序的框架
- 跳转：https://github.com/marshwallen/llm-deploy-playground/tree/main/langchain

## 😎 Intel(R) 平台 GPU 的部署
1. **Ollama 部署**
- Ollama for Intel: https://github.com/francisol/ollatel
- 该项目封装了Intel官方提供的AI推理工具包，通过图形化界面一键完成部署（Windows）
- 支持多种Intel Arc显卡架构（不包括B系列）

2. **OpenVINO 部署**
- OpenVINO™ 工具套件是 Intel 官方开源工具套件，支持计算机视觉、大型语言模型 (LLM) 和生成式 AI 等领域的 AI 开发和深度学习集成
- 跳转：https://github.com/marshwallen/llm-deploy-playground/tree/main/openvino

## 😎 Ollma + 交互式前端部署
1. **安装 Ollma**
- Ollama 是 Go 语言（主要）写的大语言模型推理框架
- 官方安装指南：https://github.com/ollama/ollama/blob/main/README.md#quickstart
- 以下为省流版（For x86 Linux）：
```sh
# Auto install
curl -fsSL https://ollama.com/install.sh | sh

# Manual install
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
sudo tar -C /usr -xzf ollama-linux-amd64.tgz
```
- 安装完毕后，在后台常驻 Ollma 服务
    - 为 Ollama 创建用户组：
    ```sh
    sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
    sudo usermod -a -G ollama $(whoami)
    ```

    - 创建 Ollama 系统服务： ```sudo vim /etc/systemd/system/ollama.service```
    ```sh
    [Unit]
    Description=Ollama Service
    After=network-online.target

    [Service]
    ExecStart=/usr/bin/ollama serve
    User=ollama
    Group=ollama
    Restart=always
    RestartSec=3
    Environment="PATH=$PATH"
    # OLLAMA_HOST 和 OLLAMA_ORIGINS 设置可保证 OLLAMA 服务能够在本机的 11434 端口上被访问
    Environment="OLLAMA_HOST=0.0.0.0:11434" 
    Environment="OLLAMA_ORIGINS=*"

    [Install]
    WantedBy=default.target
    ```
    - 重载守护进程，允许 Ollama 服务自启动
    ```sh
    sudo systemctl daemon-reload
    sudo systemctl enable ollama

    # 检查 Ollama 运行状态
    sudo systemctl status ollama.service
    ```

2. **安装 Open-Webui 交互式组件**
- 官方安装指南：https://docs.openwebui.com/
- 以下为其中一种部署方法的省流版（Docker）：
```sh
# OLLAMA_BASE_URL 改为实际指向 Ollama 服务的地址
sudo docker run -d -p 3000:8080 \
    -e OLLAMA_BASE_URL=http://ollama.server:11434 \
    -v open-webui:/app/backend/data \
    --name open-webui \
    --restart always ghcr.io/open-webui/open-webui:main

# 查看运行情况
sudo docker ps
# CONTAINER ID   IMAGE                                COMMAND           CREATED       STATUS                    PORTS                                         NAMES
# 7ca040d9cd2f   ghcr.io/open-webui/open-webui:main   "bash start.sh"   2 hours ago   Up 59 minutes (healthy)   0.0.0.0:3000->8080/tcp, [::]:3000->8080/tcp   open-webui
```
- Docker 起来后，浏览器访问 ```http://<hostname>:3000``` 即可打开交互式页面

3. **下载和部署模型**
- 在 Ollma 上下载和部署模型可以有两种方法：
    - 通过 ```ollama run``` 命令
    ```sh
    # model 名称可在 Ollama 官网上查找：https://ollama.com/
    ollama run <model>

    # 例如：模型地址：https://ollama.com/hengwen/DeepSeek-R1-Distill-Qwen-32B
    # 安装命令如下：
    ollama run hengwen/DeepSeek-R1-Distill-Qwen-32B

    # 如果希望下载某个量化模型，可以在后面加 tag，如 4bit 量化：
    ollama run hengwen/DeepSeek-R1-Distill-Qwen-32B:q4_k_m
    ```
    - 在 Open-Webui 上的可视化页面操作（设置-模型页面）

4. **其他优秀的大语言模型前端工具**
- **Lobe Chat**: https://github.com/lobehub/lobe-chat
- **AnythingLLM**: https://github.com/Mintplex-Labs/anything-llm
- **Chatbox AI**: https://github.com/Bin-Huang/Chatbox
- **NextJS Ollama LLM UI**: https://github.com/jakobhoeg/nextjs-ollama-llm-ui

## 😎 分布式推理服务部署
- 需要分布式集群管理工具（如：Kubernetes）
- 需要 1 台或多台物理机或虚拟机（master 节点），用于：
    - 部署前端页面（Open-WebUI）
    - 负载均衡和路由转发（Nginx、Ingress）
    - 模型和设备监控服务（Grafana + Prometheus + Loki）
    - 堡垒机，防御各种网络攻击（Ngx_waf）
    - 其他推理服务等
- 需要 N 台物理机或虚拟机（work 节点），用于部署推理服务实例，包括
    - 英伟达 GPU 推理后端（Ollama、Triton）
    - Intel(R) GPU 推理后端（Ollama-Intel、OpenVINO）
    - AMD GPU 推理后端
    - 国产显卡推理后端（摩尔线程、昇腾、天数智芯、沐曦）
    - CPU 推理后端
- 需要 N 台物理机（work 节点），用于数据存储：
    - 用于存放模型权重文件和模型输入输出配置（如对象存储 MinIO）
    - 用于存放 RAG 相关数据（Milvus、Elasticsearch）
    - 灾备和容灾（多副本、异地备份）
    - 用户和模型相关数据库（MySQL、PostgreSQL）或 NoSQL 数据库（如 MongoDB）
  