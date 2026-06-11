from langchain.chat_models import BaseChatModel
from pydantic import SecretStr

from yuxi.models.providers.cache import model_cache
from yuxi.utils import get_docker_safe_url
from yuxi.utils.logging_config import logger


# 这些提供商的 OpenAI 兼容流式接口，在 LangGraph v3 事件流累积工具调用时会丢失
# tool_call 的关键字段（siliconflow MiniMax 丢 name、alibaba 百炼丢 id），空值会被写入
# checkpoint，导致工具结果无法按 tool_call_id 关联、工具状态永远停留在“进行中”。
# 对这些提供商在工具调用阶段禁用流式（正文回答仍流式）以拿到完整 tool_call。
# 该缺陷属 LangChain v3 流式协议上游问题（langchain#37420 / langchainjs#10937 /
# langgraphjs#2496），截至 langchain-core 1.4.4 仍未修复，待上游修复后可移除本处理。
_NON_STREAMING_TOOL_CALL_PROVIDERS = ("siliconflow", "alibaba")


def _requires_non_streaming_tool_calls(provider_id: str, model_id: str) -> bool:
    return provider_id.startswith(_NON_STREAMING_TOOL_CALL_PROVIDERS)


def load_chat_model(fully_specified_name: str, **kwargs) -> BaseChatModel:
    if not fully_specified_name:
        raise ValueError("model spec 不能为空")

    info = model_cache.get_model_info(fully_specified_name)
    if not info:
        available_specs = model_cache.get_all_specs("chat")
        available_ids = [item.spec for item in available_specs[:10]]
        raise ValueError(
            f"Unknown model spec: '{fully_specified_name}'. "
            f"Available chat models ({len(available_specs)}): {available_ids}"
        )

    if info.model_type != "chat":
        raise ValueError(f"Model {fully_specified_name} is not a chat model (type={info.model_type})")

    api_key = info.api_key
    base_url = get_docker_safe_url(info.base_url)

    logger.debug(f"Loading model {fully_specified_name} with provider_type={info.provider_type}")

    if info.provider_type == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=info.model_id,
            api_key=SecretStr(api_key),
            base_url=base_url,
            **kwargs,
        )
    if info.provider_type == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=info.model_id,
            google_api_key=SecretStr(api_key),
            **kwargs,
        )

    from langchain_openai import ChatOpenAI

    openai_kwargs = dict(kwargs)
    if _requires_non_streaming_tool_calls(info.provider_id, info.model_id):
        openai_kwargs.setdefault("disable_streaming", "tool_calling")

    return ChatOpenAI(
        model=info.model_id,
        api_key=SecretStr(api_key),
        base_url=base_url,
        stream_usage=True,
        **openai_kwargs,
    )
