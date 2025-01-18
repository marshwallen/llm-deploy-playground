# 客户端 GRPC 流式（stream）访问 Triton 的资源
# reference: 
# 1. https://github.com/triton-inference-server/client/blob/main/src/python/examples/grpc_client.py
# 2. https://github.com/triton-inference-server/server/blob/main/qa/L0_decoupled/decoupled_test.py#L51
# 3. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/_reference/tritonclient/tritonclient.grpc.html

from functools import partial
import tritonclient.grpc as grpcclient
import numpy as np
import json
import time
import queue
import os
import base64
from tritonclient.utils import InferenceServerException

# 存储用户数据
class RecvData:
    def __init__(self):
        """
        这里用队列存放模型返回的数据
        """
        self._response_queue = queue.Queue()

class GRPCClient:
    def __init__(self, url: str, model_name="Qwen2.5-0.5B"):
        """
        初始化一些参数
        """
        self.url = url
        self.model_name = model_name

    def _callback(self, user_data, result, error):
        """
        用于 stream_infer的回调函数
        """
        if error:
            print("error: {error}\n")
            user_data._response_queue.put(error)
        else:
            user_data._response_queue.put(result)

    def _build_input_data(self, question):
        """
        构造符合输入规则的输入数据
        """
        inputs = [
            grpcclient.InferInput(name="prompt", datatype="BYTES", shape=(1,)),
            grpcclient.InferInput(name="stream", datatype="BOOL", shape=(1,)),
            grpcclient.InferInput(name="sampling_parameters", datatype="BYTES", shape=(1,))
        ]

        # 确定输入数据
        inputs[0].set_data_from_numpy(np.array([question], dtype=object))
        inputs[1].set_data_from_numpy(np.array([False], dtype=bool))
        inputs[2].set_data_from_numpy(np.array([json.dumps({
                "temperature": 0.7,
                "top_p": 0.95,
                "max_tokens": 1024
                })
            ], dtype=object))
        
        return inputs
    
    def _post_process(self, result_dict):
        """
        处理模型返回的 raw_output_contents
        """
        result = [None]*len(result_dict.keys())
        for k, v in result_dict.items():
            raw_content = "".join(["".join(x[1].get_response(as_json=True)["raw_output_contents"]) for x in v])
            # base64解码
            s = base64.b64decode(raw_content)
            s = s.decode('utf-8', errors='ignore')
            result[int(k)] = s

        return result
    
    def inference(self, questions: list, request_delay=500, verbose=False):
        """
        推理函数
        In order to run inference on a decoupled model, the client must use the bi-directional streaming RPC
        request_delay: 请求间隔(ms)
        """
        recv_data = RecvData()
 
        # 设置 grpc 客户端
        with grpcclient.InferenceServerClient(
            url=self.url, verbose=verbose
        ) as triton_client:
            # 在调用 async_stream_infer 之前，先设置好 stream
            if "TRITONSERVER_GRPC_STATUS_FLAG" in os.environ:
                metadata = {"triton_grpc_error": "true"}
                triton_client.start_stream(
                    callback=partial(self._callback, recv_data), headers=metadata
                    )
            else:
                triton_client.start_stream(
                    callback=partial(self._callback, recv_data)
                    )
                
            # 并行发送每一个请求
            for i, req in enumerate(questions):
                # 每个请求间隔一定时间
                time.sleep((request_delay / 1000))
                # inference
                triton_client.async_stream_infer(
                    model_name=self.model_name,
                    inputs=self._build_input_data(req),
                    request_id=str(i),
                )

            # 接收模型返回数据
            recv_count = 0
            result_dict = {}

            while recv_count < len(questions):
                data_item = recv_data._response_queue.get()
                # 错误处理
                if type(data_item) == InferenceServerException:
                    raise data_item
                else:
                    this_id = data_item.get_response().id
                    # 由于模型返回的数据是乱序的，所以需要将数据按顺序存入字典
                    if this_id not in result_dict:
                        result_dict[this_id] = []
                    result_dict[this_id].append((recv_count, data_item))

                recv_count += 1

        result = self._post_process(result_dict)
        return result
    
if __name__ == "__main__":
    client = GRPCClient("localhost:18998")
    
    questions = ["逻辑回归是什么？", "逻辑回归的原理是什么？", "8^2+89=?"]
    resp = client.inference(questions)

    print("="*100)
    for res in resp:
        print(res)
        print("="*10)
    
