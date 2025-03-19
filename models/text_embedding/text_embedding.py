from dify_plugin.interfaces.model.openai_compatible.text_embedding import (
    OAICompatEmbeddingModel,
)
from yarl import URL
from typing import Optional
from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult

class LLMProviderTextEmbeddingModel(OAICompatEmbeddingModel):
    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials)
        return super().validate_credentials(model, credentials)

    @staticmethod
    def _add_custom_parameters(credentials) -> None:
        credentials["endpoint_url"] = str(URL(credentials.get("endpoint_url", "https://platform.llmprovider.ai/v1")))

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        self._add_custom_parameters(credentials)
        return super()._invoke(model, credentials, texts, user, input_type)