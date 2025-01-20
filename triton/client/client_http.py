# 客户端 HTTP 访问 triton 的资源
# 支持并发测试

import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def request_test(question):
    """
    单次请求测试
    :param num_threads: 线程数量
    """
    url = "http://localhost:18999/v2/models/Qwen2.5-0.5B/generate"
    response = requests.post(url, json=get_prompt(question))

    # 检查响应状态码
    if response.status_code == 200:
        # 解析并打印响应内容
        response_data = response.json()
        return response_data['response']
    else:
        raise Exception("Error code: ", response.status_code)
    
def request_test_parallel(question, num_threads):
    """
    使用多线程并发发送请求
    :param num_threads: 线程数量
    """
    prompts = [question] * num_threads  # 根据线程数量生成相应数量的提示

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 提交所有请求任务
        futures = [executor.submit(request_test, prompt) for prompt in prompts]
        for future in as_completed(futures):
            pass  # 如果需要在任务完成后执行某些操作，可以在这里处理

def get_prompt(question):
    return {
        "prompt": question,
        "stream": False,
        "sampling_parameters": json.dumps({
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 1024
        })
    }

if __name__ == "__main__":
    # 主函数
    question = "逻辑回归是什么？"

    num_threads = 10
    # print(request_test(question))
    t0 = time.time()
    request_test_parallel(question, num_threads)
    t1 = time.time()
    print(f"并发数: {num_threads}, 耗时: {t1 - t0}(s)")