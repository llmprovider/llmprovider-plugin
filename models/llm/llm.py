import logging
from collections.abc import Generator
from typing import Optional, Union
from dify_plugin.entities.model import AIModelEntity
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.entities.model.message import (
    PromptMessage,
    PromptMessageTool,
)
from yarl import URL
from dify_plugin.errors.model import (
    InvokeError,
)
from .anthropic import AnthropicLargeLanguageModel
from .google import GoogleLargeLanguageModel

logger = logging.getLogger(__name__)

# 创建不同厂商的模型实例
model_schemas = []
anthropic_llm = AnthropicLargeLanguageModel(model_schemas)
google_llm = GoogleLargeLanguageModel(model_schemas)
logger = logging.getLogger(__name__)


class LLMProviderLargeLanguageModel(OAICompatLargeLanguageModel):
    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = str(URL(credentials.get("endpoint_url", "https://platform.llmprovider.ai/v1")))
        credentials["mode"] = self.get_model_mode(model).value
        credentials["function_calling_type"] = "tool_call"

    def _dispatch_to_appropriate_model(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None
    ) -> Union[LLMResult, Generator]:
        """根据模型名称分发到适当的模型处理类"""
        # 检查模型名称是否以 "claude" 开头
        if model.startswith("claude"):
            return anthropic_llm._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)
                # 检查模型名称是否以 "gemini" 开头且不以 "-nothink" 或 "-search" 结尾
        if model.startswith("gemini") and not (model.endswith("-nothink") or model.endswith("-search")):
            return google_llm._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

        # 默认使用父类的生成方法
        return super()._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def _invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        try:
            self._update_credential(model, credentials)
            enable_thinking = model_parameters.pop("enable_thinking", None)
            if enable_thinking is not None:
                model_parameters["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

            return self._dispatch_to_appropriate_model(
                model, credentials, prompt_messages, model_parameters, tools, stop, stream, user
            )
        except Exception as e:
            # 记录异常信息
            logger.error(f"Error invoking model {model}: {str(e)}")

            # 根据异常类型映射到统一的错误类型
            for error_type, exception_types in self._invoke_error_mapping.items():
                if any(isinstance(e, exc_type) for exc_type in exception_types):
                    raise error_type(str(e))

            # 如果没有匹配的错误类型，则抛出原始异常
            raise InvokeError(f"Unexpected error: {str(e)}")

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._update_credential(model, credentials)
        return super().validate_credentials(model, credentials)

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        self._update_credential(model, credentials)
        return super().get_customizable_model_schema(model, credentials)

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._update_credential(model, credentials)
        return super().get_num_tokens(model, credentials, prompt_messages, tools)
