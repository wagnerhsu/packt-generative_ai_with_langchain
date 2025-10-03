# math_agent.py
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.tools import Tool

# =======================
# Step 1: 定义状态（State）
# =======================

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # 消息列表，自动累加
    current_step: str                        # 当前步骤
    should_retry: bool                       # 是否需要重试

# =======================
# Step 2: 定义工具（Tools）
# =======================

def calculator_tool(expression: str) -> str:
    """
    简易计算器工具（实际项目可替换为 WolframAlpha 或 Python REPL）
    """
    try:
        # 安全起见，仅允许基本运算
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "❌ 表达式包含非法字符"
        result = eval(expression, {"__builtins__": {}})  # 安全执行
        return str(result)
    except Exception as e:
        return f"❌ 计算失败: {e}"

tool = Tool(
    name="calculator",
    description="用于执行数学表达式计算，输入如: 5 - 2 + 3",
    func=calculator_tool,
)

tools = [tool]
tool_names = {t.name for t in tools}

# =======================
# Step 3: 定义节点（Nodes）
# =======================

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key="your-openai-api-key"  # 替换为你的 API Key
)

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools(tools)

def decide_action(state: AgentState) -> AgentState:
    """
    决策节点：让 LLM 决定是回答还是调用工具
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "current_step": "action",
        "should_retry": False
    }

def call_tool(state: AgentState) -> AgentState:
    """
    工具调用节点
    """
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return {
            "messages": [AIMessage(content="未检测到工具调用。")],
            "current_step": "end",
            "should_retry": False
        }

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]

    if tool_name not in tool_names:
        result = f"❌ 未知工具: {tool_name}"
    else:
        tool_func = tool.func
        result = tool_func(tool_call["args"])

    tool_response = ToolMessage(
        content=result,
        name=tool_name,
        tool_call_id=tool_call["id"]
    )
    return {"messages": [tool_response], "current_step": "tool_response", "should_retry": False}

def finalize_answer(state: AgentState) -> AgentState:
    """
    最终回答节点
    """
    # 再次调用 LLM 生成自然语言回答
    final_response = llm.invoke(state["messages"])
    return {
        "messages": [final_response],
        "current_step": "end",
        "should_retry": False
    }

# =======================
# Step 4: 构建图（Graph）
# =======================

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("decide_action", decide_action)
workflow.add_node("call_tool", call_tool)
workflow.add_node("finalize_answer", finalize_answer)

# 设置入口
workflow.set_entry_point("decide_action")

# 添加边（Edges）
workflow.add_conditional_edges(
    "decide_action",
    lambda x: "call_tool" if x["messages"][-1].tool_calls else "finalize_answer",
    {
        "call_tool": "call_tool",
        "finalize_answer": "finalize_answer"
    }
)

# 工具调用后回到决策节点（支持多次调用）
workflow.add_edge("call_tool", "decide_action")
workflow.add_edge("finalize_answer", END)

# 编译图
app = workflow.compile()

# =======================
# Step 5: 运行示例
# =======================

if __name__ == "__main__":
    print("🧠 数学问题解决Agent已启动（输入 'quit' 退出）\n")

    while True:
        question = input("📌 请输入数学问题: ").strip()
        if question.lower() in ["quit", "exit"]:
            print("👋 再见！")
            break
        if not question:
            continue

        try:
            result = app.invoke({
                "messages": [HumanMessage(content=question)],
                "current_step": "start",
                "should_retry": False
            })

            # 输出最终答案
            final_message = result["messages"][-1]
            print(f"\n✅ 答案: {final_message.content}\n")

            # （可选）打印完整消息流（调试用）
            # for m in result["messages"]:
            #     print(f"[{m.type.upper()}]: {m.content}")

        except Exception as e:
            print(f"❌ 执行出错: {e}")