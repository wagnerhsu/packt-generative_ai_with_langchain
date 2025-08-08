from openai import OpenAI

# 创建客户端，指向 LMStudio 的本地服务器
client = OpenAI(
    base_url="http://localhost:1234/v1",  # LMStudio 默认地址
    api_key="not-needed"  # LMStudio 不需要 API key，但必须传参
)

# 发送请求
response = client.chat.completions.create(
    model="local-model",  # 这个值其实会被忽略，实际使用的是 LMStudio 中加载的模型
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "请用中文介绍一下你自己。"}
    ],
    temperature=0.7,
    max_tokens=256
)

# 输出响应
print(response.choices[0].message.content)