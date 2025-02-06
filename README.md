# LLM-deploy-playground
- LLM 本地部署实践

## Introduction
- 推理服务部署参考：https://github.com/liguodongiot/llm-action

## 推理框架的基本要素
1. 得有一个部署平台/集群（如：Kubernetes）。用于对一个推理服务的生命周期的管理（模型的加载/卸载，模型服务实例的弹性扩缩容等）
2. 还应该有一个负载均衡器（如：Ingress），解决模型推理服务实例负载均衡的问题。将客户端的请求，均衡的分配给推理服务实例
3. 还有一个模型仓库（如：本地文件系统），对模型权重文件和模型输入输出配置进行管理
4. 之后还应该有一个模型监控服务，用于观察模型的宏观/微观的数据
5. 最后，也是最核心的，它还应该有一个推理服务，进行模型推理

## 1 Triton Inference Server 部署
- 跳转：https://github.com/marshwallen/llm-deploy-playground/tree/main/triton

## 2 LangChain 部署
- 跳转：https://github.com/marshwallen/llm-deploy-playground/tree/main/langchain

## 3 Intel(R) 平台 GPU 的部署
- 跳转：https://github.com/marshwallen/llm-deploy-playground/tree/main/openvino

## 4 Ollma + Open-Webui 交互式页面部署
1. **安装 Ollma**
- 官方安装指南：https://github.com/ollama/ollama/blob/main/README.md#quickstart
- 以下为省流版：
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

4. **愉快地玩起来吧**