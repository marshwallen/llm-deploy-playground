# 实现一个 chatbot 后端
from local_llm import LocalLangchainLLM
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

class Bot:
    def __init__(self):
        """
        支持 Message persistence 的聊天 Bot
        """
        # 初始化 LLM
        self.model = LocalLangchainLLM("Qwen/Qwen2.5-3B")

        # 创建一个工作流图以实现多轮对话
        workflow = StateGraph(state_schema=MessagesState)

        # 定义模型调用函数
        def call_model(state: MessagesState):
            response = self.model.invoke(state["messages"])
            return {"messages": response}

        # 定义图的点和边
        workflow.add_edge(START, "model")  
        workflow.add_node("model", call_model)

        # 添加记忆功能
        memory = MemorySaver()   
        self.app = workflow.compile(checkpointer=memory) # 使工作流能够在运行中保存和恢复状态

    def chat(self, query, config):
        """
        实现聊天逻辑
        每一个聊天窗口都有一个对应的pid, 表示了不同的上下文
        config = {"configurable": {"thread_id": "abc123"}}
        """
        print(f"@@ You [t_id: {config["configurable"]["thread_id"]}]: {query}")
        input_messages = [HumanMessage(query)]
        output = self.app.invoke({"messages": input_messages}, config)
        print(f"@@ AI [t_id: {config["configurable"]["thread_id"]}]: {output["messages"][-1].content}")

if __name__ == "__main__":
    bot = Bot()
    t0 = {"configurable": {"thread_id": "t0"}}
    t1 = {"configurable": {"thread_id": "t1"}}
    bot.chat("我会告诉你我今天吃了两个馒头和一包榨菜，还有两个鸡蛋", t0)
    bot.chat("我今天吃了什么", t0)
    bot.chat("我今天吃了什么", t1)
    