# langchain-openai-thinking-agui

为 LangChain 的 Chat 模型补充思考内容（reasoning/thinking）提取能力，输出格式与 [ag-ui-langgraph](https://github.com/ag-ui-protocol/ag-ui) 兼容。

支持所有 OpenAI 兼容接口的思考模型，包括 DeepSeek、Kimi 等。

## 安装

```bash
pip install langchain-openai-thinking-agui
```

## 使用

```python
from langchain_openai_thinking_agui import ChatOpenAIWithThinking

model = ChatOpenAIWithThinking(
    model="deepseek-reasoner",
    base_url="https://api.deepseek.com/v1",
    api_key="your-api-key",
)

res = model.invoke("解释一下天空是蓝色的")
# res.content[0]["thinking"] 是思考内容
# res.content[1]["text"]    是正式回答
```

## 输出格式

思考内容以 Anthropic content blocks 格式存放在 `AIMessage.content` 中，ag-ui-langgraph 可自动识别并触发 `THINKING_*` 事件流：

```python
[
    {"type": "text", "thinking": "<思考过程>", "index": 0},
    {"type": "text", "text": "<正式回答>"},
]
```

## License

MIT
