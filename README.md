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
- https://github.com/marshwallen/llm-deploy-playground/tree/main/triton

## 2 LangChain 部署
- https://github.com/marshwallen/llm-deploy-playground/tree/main/langchain