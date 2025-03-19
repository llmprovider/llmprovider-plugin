import logging
from collections.abc import Generator
from typing import Optional, Union
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from yarl import URL
from dify_plugin import OAICompatLargeLanguageModel

logger = logging.getLogger(__name__)

class LLMProviderLargeLanguageModel(OAICompatLargeLanguageModel):
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
        self._update_credential(self, model, credentials)
        return super()._invoke(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._update_credential(self, model, credentials)
        return super().validate_credentials(model, credentials)

    @staticmethod
    def _update_credential(self, model: str, credentials: dict):
        credentials["endpoint_url"] = str(URL(credentials.get("endpoint_url", "https://platform.llmprovider.ai/v1")))
        credentials["mode"] = self.get_model_mode(model).value