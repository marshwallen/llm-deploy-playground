from fastapi import FastAPI
from pydantic import BaseModel
from openvino_model import OpenVINOModel
import uvicorn

# 初始化 FastAPI
app = FastAPI()

# 加载模型
model = OpenVINOModel(
    model_id="Qwen/Qwen2.5-0.5B",
    device="GPU"
)

# 定义请求体
class PredictRequest(BaseModel):
    # 根据模型输入格式定义
    system_prompt: str
    user_prompt: str  

# 定义 API 路由
@app.post("/predict")
def predict(request: PredictRequest):
    """
    调用模型推理
    """
    message = [
        {
            "role": "system",
            "content": request.system_prompt
        },
        {
            "role": "user",
            "content": request.user_prompt
        }
    ]
    output = model.predict(message)
    return {"output": output}

# 运行 API
if __name__ == "__main__":
    # 使用 uvicorn 运行 API
    uvicorn.run(app, host="0.0.0.0", port=8000)