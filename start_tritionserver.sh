docker run --gpus all --rm \
    -p 18999:8000 -p 18998:8001 -p 18997:8002 \
    --shm-size=1G -e PYTHONIOENCODING=utf-8 \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ${PWD}/model_repository/:/models tritonserver:24.12-vllm-py3-custom \
    tritonserver --model-repository=/models \
    --model-control-mode explicit \
    --load-model Qwen2.5-0.5B

