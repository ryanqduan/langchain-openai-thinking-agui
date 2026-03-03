"""
LangChain chat model 子类，将思考模型的 reasoning_content 转换为
ag_ui_langgraph 可识别的 Anthropic content blocks 格式。

ag_ui_langgraph 的 resolve_reasoning_content() 检测规则：
  content[0]["thinking"] → 思考内容
  其余 block 的 "text"  → 正式回答
"""
from typing import Any, Optional

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI as _ChatOpenAIBase

# ── 共享工具函数 ──────────────────────────────────────────────────────────────

def _build_thinking_content(reasoning: str, text: str) -> list:
    """构建 ag_ui_langgraph 可识别的 Anthropic content blocks。"""
    return [
        {"type": "text", "thinking": reasoning, "index": 0},
        {"type": "text", "text": text},
    ]


def _extract_reasoning(msg_dict: dict) -> Optional[str]:
    """从 message 字典中提取思考内容（兼容 reasoning_content / thinking_content 字段名）。"""
    return msg_dict.get("reasoning_content") or msg_dict.get("thinking_content")


# ── ChatOpenAIWithThinking ────────────────────────────────────────────────────

class ChatOpenAIWithThinking(_ChatOpenAIBase):
    """ChatOpenAI 子类，支持 OpenAI 兼容接口的思考模型（DeepSeek、Kimi 等）。

    只覆盖两个方法：
      - _create_chat_result                ：非流式，在 super() 结果上补充思考内容
      - _convert_chunk_to_generation_chunk ：流式，拦截含 reasoning_content 的 delta
    """

    def _create_chat_result(
        self,
        response: Any,
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        # 先由父类完成常规字段的解析
        result = super()._create_chat_result(response, generation_info)

        response_dict = response if isinstance(response, dict) else response.model_dump()
        for i, res in enumerate(response_dict.get("choices", [])):
            reasoning = _extract_reasoning(res.get("message", {}))
            if not reasoning:
                continue
            # 将父类生成的 AIMessage 替换为带 thinking blocks 的版本
            orig: AIMessage = result.generations[i].message  # type: ignore[assignment]
            result.generations[i].message = AIMessage(
                content=_build_thinking_content(reasoning, orig.content),  # type: ignore[arg-type]
                additional_kwargs=orig.additional_kwargs,
                usage_metadata=orig.usage_metadata,
                id=orig.id,
            )

        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        # 先由父类处理常规逻辑
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        # 检查 delta 中是否含思考内容
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices", [])
        if not choices:
            return gen_chunk

        reasoning_delta = (choices[0].get("delta") or {}).get("reasoning_content")
        if not reasoning_delta:
            return gen_chunk

        # 将普通 chunk 替换为思考 chunk，保留父类已解析的 generation_info
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=[{"type": "text", "thinking": reasoning_delta, "index": 0}]
            ),
            generation_info=gen_chunk.generation_info if gen_chunk else None,
        )
