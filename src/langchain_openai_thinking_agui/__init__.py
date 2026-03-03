"""
langchain-openai-thinking-agui

为 LangChain 的 Chat 模型补充思考内容提取能力，
输出格式兼容 ag_ui_langgraph 的 Anthropic content blocks 规范。
"""

from langchain_openai_thinking_agui.chat_models import ChatOpenAIWithThinking

__version__ = "0.1.0"
__all__ = ["ChatOpenAIWithThinking"]
