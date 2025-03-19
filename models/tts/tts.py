from dify_plugin.interfaces.model.openai_compatible.tts import OAICompatText2SpeechModel
from yarl import URL
from typing import Optional
from collections.abc import Generator

class OpenAIText2SpeechModel(OAICompatText2SpeechModel):
    def validate_credentials(self, model: str, credentials: dict) -> None:
        self._add_custom_parameters(credentials)
        return super().validate_credentials(model, credentials)

    @staticmethod
    def _add_custom_parameters(credentials) -> None:
        credentials["endpoint_url"] = str(URL(credentials.get("endpoint_url", "https://platform.llmprovider.ai/v1")))

    def _invoke(
        self,
        model: str,
        tenant_id: str,
        credentials: dict,
        content_text: str,
        voice: str,
        user: Optional[str] = None,
    ) -> Generator[bytes, None, None]:
        self._add_custom_parameters(credentials)
        return super()._invoke(model, tenant_id, credentials, content_text, voice, user)