curl -X POST localhost:18999/v2/models/Qwen2.5-0.5B/generate \
-d '{"prompt": "逻辑回归是什么?", "stream": false, "sampling_parameters": "{\"temperature\": 0.7, \"top_p\": 0.95, \"max_tokens\": 1024}"}'


