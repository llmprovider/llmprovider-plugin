import logging
from collections.abc import Generator
from typing import Optional, Union
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from yarl import URL
from dify_plugin import OAICompatLargeLanguageModel
from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from .anthropic import AnthropicLargeLanguageModel
from .google import GoogleLargeLanguageModel

logger = logging.getLogger(__name__)

# 创建不同厂商的模型实例
model_schemas = []
anthropic_llm = AnthropicLargeLanguageModel(model_schemas)
google_llm = GoogleLargeLanguageModel(model_schemas)

class LLMProviderLargeLanguageModel(OAICompatLargeLanguageModel):
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
        
        # 检查模型名称是否以 "gemini" 开头
        if model.startswith("gemini"):
            return google_llm._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)
        
        # 默认使用父类的生成方法
        if stream:
            model_parameters["stream_options"] = {
                "include_usage": True,
            }
        return super()._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

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
            self._update_credential(self, model, credentials)
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
        self._update_credential(self, model, credentials)
        
        # 根据模型类型分发到对应的处理类
        if model.startswith("claude"):
            return anthropic_llm.validate_credentials(model, credentials)
        elif model.startswith("gemini"):
            return google_llm.validate_credentials(model, credentials)
        else:
            return super().validate_credentials(model, credentials)

    @staticmethod
    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = str(URL(credentials.get("endpoint_url", "https://platform.llmprovider.ai/v1")))
        credentials["mode"] = self.get_model_mode(model).value
        credentials["function_calling_type"] = "tool_call"

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        self._update_credential(self, model, credentials)
        
        # 根据模型类型分发到对应的处理类
        if model.startswith("claude"):
            return anthropic_llm.get_num_tokens(model, credentials, prompt_messages, tools)
        elif model.startswith("gemini"):
            return google_llm.get_num_tokens(model, credentials, prompt_messages, tools)
        else:
            return super().get_num_tokens(model, credentials, prompt_messages, tools)