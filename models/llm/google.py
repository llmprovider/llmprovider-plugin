import base64
import logging
import json
import uuid
from collections.abc import Generator, Iterator, Sequence
from typing import Optional, Union, Mapping, Any

import requests
from google import genai
from google.genai import errors, types
from dify_plugin.entities.model.llm import (
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    MultiModalPromptMessageContent,
    PromptMessageContent,
    AudioPromptMessageContent,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    ImagePromptMessageContent,
    TextPromptMessageContent,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from httpx import Timeout



class GoogleLargeLanguageModel(LargeLanguageModel):
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
    ) -> Union[LLMResult, Generator[LLMResultChunk]]:
        """
        Invoke large language model

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        _ = user
        return self._generate(
            model,
            credentials,
            prompt_messages,
            model_parameters,
            tools,
            stop,
            stream,
            user,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return:md = genai.GenerativeModel(model)
        """
        prompt = self._convert_messages_to_prompt(prompt_messages)
        return self._get_num_tokens_by_gpt2(prompt)

    def _convert_messages_to_prompt(self, messages: list[PromptMessage]) -> str:
        """
        Format a list of messages into a full prompt for the Google model

        :param messages: List of PromptMessage to combine.
        :return: Combined string with necessary human_prompt and ai_prompt tags.
        """
        messages = messages.copy()
        text = "".join(
            (self._convert_one_message_to_text(message) for message in messages)
        )
        return text.rstrip()

    def _convert_tools_to_glm_tool(self, tools: list[PromptMessageTool]) -> types.Tool:
        """
        Convert tool messages to google-genai's Tool Type.

        :param tools: tool messages
        :return: glm tools
        """
        function_declarations = []
        for tool in tools:
            properties = {}
            for key, value in tool.parameters.get("properties", {}).items():
                property_def = {
                    "type": "STRING",
                    "description": value.get("description", ""),
                }
                if "enum" in value:
                    property_def["enum"] = value["enum"]
                properties[key] = property_def

            if properties:
                parameters = types.Schema(
                    type="OBJECT",
                    properties=properties,
                    required=tool.parameters.get("required", []),
                )
            else:
                parameters = None

            functions = types.FunctionDeclaration(
                name=tool.name,
                parameters=parameters,
                description=tool.description,
            )
            function_declarations.append(functions)

        return types.Tool(function_declarations=function_declarations)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials

        :param model: model name
        :param credentials: model credentials
        :return:
        """
        try:
            ping_message = UserPromptMessage(content="ping")
            self._generate(
                model=model,
                credentials=credentials,
                prompt_messages=[ping_message],
                stream=False,
                model_parameters={"max_output_tokens": 5},
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))
        
    def _get_response_modalities(self, model: str) -> list[str]:
        """_get_response_modalities returns response modalities supported 
        by the given model.
        """
        # FIXME(QuantumGhost): Multimodal output is currently limited to 
        # the gemini-2.0-flash-experiment model. The model name is currently 
        # hardcoded for simplicity; consider revisiting this approach for flexibility.
        if model != "gemini-2.0-flash-exp":
            return ["Text"]
        
        return ["Text", "Image"]

    def _to_credential_kwargs(
            self, credentials: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """
        Transform credentials to kwargs for model instance

        :param credentials:
        :return:
        """
        credentials_kwargs = {
            "api_key": credentials["api_key"],
            "base_url": "https://platform.llmprovider.ai",
            "timeout": Timeout(615.0, read=600.0, write=10.0, connect=5.0),
            "max_retries": 1,
        }
        api_url = credentials.get("endpoint_url")
        if api_url:
            credentials_kwargs["base_url"] = api_url.removesuffix("/v1").rstrip("/")

        return credentials_kwargs


    def _generate(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: Mapping[str, Any],
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator[LLMResultChunk]]:
        conditions: list[bool] = [
            bool(model_parameters.get("json_schema")),
            bool(model_parameters.get("grounding")),
            bool(tools),
        ]
        if sum(conditions) >= 2:
            raise InvokeError(
                "gemini not support use multiple features at same time: json_schema, grounding, tools+knowledge"
            )
        config = types.GenerateContentConfig()
        if system_instruction := self._get_system_instruction(prompt_messages=prompt_messages):
            config.system_instruction = system_instruction
        if schema := model_parameters.get("json_schema"):
            try:
                schema = json.loads(schema)
            except (TypeError, ValueError) as exc:
                raise InvokeError("Invalid JSON Schema") from exc
            config.response_schema = schema
            config.response_mime_type = "application/json"
        else:
            # Enable multimodal support only if JSON schema is not provided.
            config.response_modalities = self._get_response_modalities(model)

        if stop:
            config.stop_sequences = stop
        
        config.top_p = model_parameters.get("top_p", None)
        config.top_k = model_parameters.get("top_k", None)
        config.temperature = model_parameters.get("temperature", None)
        config.max_output_tokens = model_parameters.get("max_output_tokens", None)

        config.tools = []
        if model_parameters.get("grounding"):
            config.tools.append(types.Tool(google_search=types.GoogleSearch()))
        if tools:
            config.tools.append(self._convert_tools_to_glm_tool(tools))

        credentials_kwargs = self._to_credential_kwargs(credentials)
        genai_client = genai.Client(api_key=credentials_kwargs["api_key"], http_options={
            "base_url": credentials_kwargs["base_url"],
        })

        history = []
        file_server_url_prefix = credentials.get("file_url") or None
        for msg in prompt_messages:  # makes message roles strictly alternating
            content = self._format_message_to_glm_content(
                msg,
                file_server_url_prefix=file_server_url_prefix)
            if history and history[-1].role == content.role:
                history[-1].parts.extend(content.parts)
            else:
                history.append(content)
        # contents = self._convert_to_contents(prompt_messages=prompt_messages)

        if stream:
            response = genai_client.models.generate_content_stream(
                model=model,
                contents=history,
                config=config,
            )
            return self._handle_generate_stream_response(model, credentials, response, prompt_messages)

        response = genai_client.models.generate_content(
            model=model,
            contents=history,
            config=config,
        )
        return self._handle_generate_response(model, credentials, response, prompt_messages)

    def _convert_one_message_to_text(self, message: PromptMessage) -> str:
        """
        Convert a single message to a string.

        :param message: PromptMessage to convert.
        :return: String representation of the message.
        """
        human_prompt = "\n\nuser:"
        ai_prompt = "\n\nmodel:"
        content = message.content
        if isinstance(content, list):
            content = "".join((c.data for c in content if c.type != PromptMessageContentType.IMAGE))
        if isinstance(message, UserPromptMessage):
            message_text = f"{human_prompt} {content}"
        elif isinstance(message, AssistantPromptMessage):
            message_text = f"{ai_prompt} {content}"
        elif isinstance(message, SystemPromptMessage | ToolPromptMessage):
            message_text = f"{human_prompt} {content}"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _get_system_instruction(
        self, *, prompt_messages: Sequence[PromptMessage]
    ) -> str:
        # `prompt_messages` should be a sequence containing at least one 
        # `SystemPromptMessage`.
        # If the sequence is empty or the first element is not a 
        # `SystemPromptMessage`, 
        # the method returns an empty string, effectively indicating the 
        # absence of a system instruction.
        if len(prompt_messages) == 0:
            return ""
        if not isinstance(prompt_messages[0], SystemPromptMessage):
            return ""
        system_instruction = ""
        prompt = prompt_messages[0]
        if isinstance(prompt.content, str):
            system_instruction = prompt.content
        elif isinstance(prompt.content, list):
            system_instruction = ""
            for content in prompt.content:
                if isinstance(content, TextPromptMessageContent):
                    system_instruction += content.data
                else:
                    raise InvokeError(
                        "system prompt content does not support image, document, video, audio"
                    )
        else:
            raise InvokeError("system prompt content must be a string or a list of strings")
        return system_instruction

    def _format_message_to_glm_content(
            self, message: PromptMessage, 
            file_server_url_prefix: str|None=None,
        ) -> types.Content:
        """
        Format a single message into glm.Content for Google API

        :param message: one PromptMessage
        :return: glm Content representation of message
        """
        if isinstance(message, UserPromptMessage):
            glm_content = types.Content(role="user", parts=[])
            if isinstance(message.content, str):
                glm_content.parts.append(types.Part.from_text(text=message.content))
            else:
                for c in message.content:
                    if c.type == PromptMessageContentType.TEXT:
                        glm_content.parts.append(types.Part.from_text(text=c.data))
                    else:
                        # 使用base64数据而不是上传文件
                        part = self._create_part_from_content(
                            message_content=c, file_server_url_prefix=file_server_url_prefix
                        )
                        glm_content.parts.append(part)

            return glm_content
        elif isinstance(message, AssistantPromptMessage):
            glm_content = types.Content(role="model", parts=[])
            if message.content:
                glm_content.parts.append(types.Part.from_text(text=message.content))
            if message.tool_calls:
                glm_content.parts.append(
                    types.Part.from_function_call(
                        name=message.tool_calls[0].function.name,
                        args=json.loads(message.tool_calls[0].function.arguments),
                    )
                )
            return glm_content
        elif isinstance(message, SystemPromptMessage):
            return types.Content(role="user", parts=[types.Part.from_text(text=message.content)])
        elif isinstance(message, ToolPromptMessage):
            return types.Content(
                role="function",
                parts=[types.Part.from_function_response(name=message.name, response={"response": message.content})],
            )
        else:
            raise ValueError(f"Got unknown type {message}")

    def _create_part_from_content(
        self, message_content: MultiModalPromptMessageContent,
        file_server_url_prefix: str | None = None,
    ) -> types.Part:
        """
        从多模态内容创建Part，使用base64数据而不是上传文件
        
        :param message_content: 多模态消息内容
        :param file_server_url_prefix: 文件服务器URL前缀（当需要从URL获取内容时）
        :return: Gemini API的Part对象
        """
        # 如果有base64数据，直接使用
        if message_content.base64_data:
            file_data = base64.b64decode(message_content.base64_data)
            return types.Part.from_bytes(
                data=file_data,
                mime_type=message_content.mime_type
            )
        
        # 如果有URL，从URL下载数据并转换为base64
        if message_content.url:
            try:
                file_url = message_content.url
                if file_server_url_prefix:
                    file_url = f"{file_server_url_prefix.rstrip('/')}/files{message_content.url.split('/files')[-1]}"
                
                if not file_url.startswith(("https://", "http://")):
                    raise ValueError("需要设置 FILES_URL 环境变量！")
                
                response = requests.get(file_url)
                response.raise_for_status()
                
                return types.Part.from_bytes(
                    data=response.content,
                    mime_type=message_content.mime_type
                )
            except Exception as ex:
                raise ValueError(f"从URL获取数据失败 {file_url}: {ex}")
        
        raise ValueError("消息内容必须包含base64_data或url")

    def _handle_generate_response(
            self,
            model: str,
            credentials: dict,
            response: types.GenerateContentResponse,
            prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        """
        Handle llm response

        :param model: model name
        :param credentials: credentials
        :param response: response
        :param prompt_messages: prompt messages
        :return: llm response
        """
        # transform assistant message to prompt message
        assistant_prompt_message = AssistantPromptMessage(content=response.text)

        # calculate num tokens
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            if prompt_tokens is None:
                raise ValueError("prompt_tokens is None")
            completion_tokens = response.usage_metadata.candidates_token_count
            if completion_tokens is None:
                raise ValueError("completion_tokens is None")
        else:
            prompt_tokens = self.get_num_tokens(model, credentials, prompt_messages)
            completion_tokens = self.get_num_tokens(model, credentials, [assistant_prompt_message])

        # transform usage
        # copy credentials to avoid modifying the original dict
        usage = self._calc_response_usage(model, dict(credentials), prompt_tokens, completion_tokens)

        # transform response
        result = LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=assistant_prompt_message,
            usage=usage,
        )

        return result

    def _handle_generate_stream_response(
            self,
            model: str,
            credentials: dict,
            response: Iterator[types.GenerateContentResponse],
            prompt_messages: list[PromptMessage],
    ) -> Generator[LLMResultChunk]:
        """
        Handle llm stream response

        :param model: model name
        :param credentials: credentials
        :param response: response
        :param prompt_messages: prompt messages
        :return: llm response chunk generator result
        """
        index = -1
        prompt_tokens = 0
        completion_tokens = 0
        for chunk in response:
            if not chunk.candidates:
                continue
            for candidate_idx, candidate in enumerate(chunk.candidates):
                if not candidate.content or not candidate.content.parts:
                    continue
                try:
                    message = self._parse_parts(candidate.content.parts)
                except Exception as e:
                    raise e
                index += len(candidate.content.parts)
                if chunk.usage_metadata:
                    prompt_tokens += chunk.usage_metadata.prompt_token_count or 0
                    completion_tokens += (
                        chunk.usage_metadata.candidates_token_count or 0
                    )

                # if the stream is not finished, yield the chunk
                if not candidate.finish_reason:
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=list(prompt_messages),
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=message,
                        ),
                    )
                # if the stream is finished, yield the chunk and the finish reason
                else:
                    if prompt_tokens == 0 or completion_tokens == 0:
                        prompt_tokens = self.get_num_tokens(
                            model=model,
                            credentials=credentials,
                            prompt_messages=prompt_messages,
                        )
                        completion_tokens = self.get_num_tokens(
                            model=model,
                            credentials=credentials,
                            prompt_messages=[message],
                        )
                    usage = self._calc_response_usage(
                        model=model,
                        credentials=dict(credentials),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=list(prompt_messages),
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=message,
                            finish_reason=candidate.finish_reason,
                            usage=usage,
                        ),
                    )

    def _render_grounding_source(self, grounding_metadata: types.GroundingMetadata) -> str:
        """
        Render google search source links
        """
        result = "\n\n**Search Sources:**\n"
        for index, entry in enumerate(grounding_metadata.grounding_chunks, start=1):
            result += f"{index}. [{entry.web.title}]({entry.web.uri})\n"
        return result
                

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        """
        return {
            InvokeConnectionError: [
                errors.APIError, 
                errors.ClientError,
            ],
            InvokeServerUnavailableError: [
                errors.ServerError,
            ],
            InvokeBadRequestError: [
                errors.ClientError,
                errors.UnknownFunctionCallArgumentError,
                errors.UnsupportedFunctionError,
                errors.FunctionInvocationError,
            ],
        }

    def _parse_parts(self, parts: Sequence[types.Part], /) -> AssistantPromptMessage:
        contents: list[PromptMessageContent] = []
        function_calls = []
        for i, part in enumerate(parts):
            if part.text:
                contents.append(TextPromptMessageContent(data=part.text))
            if part.function_call:
                function_call = part.function_call
                function_call_id = function_call.id
                function_call_name = function_call.name
                function_call_args = function_call.args
                # 尝试转换 function_call_id 为字符串
                if not isinstance(function_call_id, str):
                    if function_call_id is None:
                        print(f"function_call_id 是 None，生成随机ID")
                        function_call_id = str(uuid.uuid4())
                    else:
                        function_call_id = str(function_call_id)

                if not isinstance(function_call_name, str):
                    print(f"    错误: function_call_name 不是字符串: {type(function_call_name)}")
                    raise InvokeError(f"function_call_name received is not a string, got {type(function_call_name)}: {repr(function_call_name)}")

                if not isinstance(function_call_args, dict):
                    print(f"    错误: function_call_args 不是字典: {type(function_call_args)}")
                    raise InvokeError(f"function_call_args received is not a dict, got {type(function_call_args)}: {repr(function_call_args)}")

                print(f"    最终使用的id: {repr(function_call_id)}")

                function_call = AssistantPromptMessage.ToolCall(
                    id=function_call_id,
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=function_call_name,
                        arguments=json.dumps(function_call_args),
                    ),
                )
                function_calls.append(function_call)
            if part.inline_data:
                inline_data = part.inline_data
                mime_type = inline_data.mime_type
                data = inline_data.data
                if mime_type is None:
                    raise InvokeError("receive inline_data with no mime_type")
                if data is None:
                    raise InvokeError("receive inline_data with no data")
                if mime_type.startswith("image/"):
                    mime_subtype = mime_type.split("/", maxsplit=1)[-1]
                    # Here the data returned by genai-sdk is already a base64-encoded
                    # byte string, so just decode it to utf-8 string is enough.
                    contents.append(
                        ImagePromptMessageContent(
                            format=mime_subtype,
                            base64_data=base64.b64encode(data).decode(),
                            mime_type=mime_type,
                        )
                    )
                else:
                    raise InvokeError(f"unsupported mime_type {mime_type}")

        # FIXME: This is a workaround to fix the typing issue in the dify_plugin
        # https://github.com/langgenius/dify-plugin-sdks/issues/41
        # fixed_contents = [content.model_dump(mode="json") for content in contents]
        message = AssistantPromptMessage(
            content=contents, tool_calls=function_calls  # type: ignore
        )
        return message
    
    def _convert_to_contents(self, prompt_messages: Sequence[PromptMessage]) -> list[types.ContentDict]:
        """_convert_to_content convert input prompt messages to contents sent to Google 
        Gemini models.
        """
        contents: list[types.ContentDict] = []

        last_content: Optional[types.ContentDict] = None
        for prompt in prompt_messages:
            # skip all `SystemPromptMessage` messages, since they are handled 
            # by `_get_system_instruction`.
            if isinstance(prompt, SystemPromptMessage):
                continue

            content = self._convert_to_content_dict(prompt)
            if last_content is None:
                last_content = content
                continue
            
            if last_content.get('role') != content.get('role'):
                contents.append(last_content)
                last_content = content
                continue
            # merge parts with the same role.
            parts = last_content.get("parts") or []
            parts.extend(content.get("parts") or [])
            last_content = types.ContentDict(
                parts=parts,
                role=content.get("role"),
            )
        # append the last content if exists
        if last_content is not None:
            contents.append(last_content)
        return contents

    def _convert_to_content_dict(self, prompt: PromptMessage) -> types.ContentDict:
        
        role = "user" if isinstance(prompt, UserPromptMessage) else "model"
        parts = []

        prompt_contents = prompt.content
        if isinstance(prompt_contents, str):
            parts.append(types.Part.from_text(text=prompt_contents))
        elif isinstance(prompt_contents, list):
            for content in prompt_contents:
                parts.append(_content_to_part(content))
        else:
            raise InvokeError("prompt content must be a string or a list")
        
        return types.ContentDict(parts=parts, role=role)


def _content_to_part(content: PromptMessageContent) -> types.Part:
    if isinstance(content, TextPromptMessageContent):
        return types.Part.from_text(text=content.data)
    elif isinstance(
                    content,
                    (ImagePromptMessageContent,
                    DocumentPromptMessageContent,
                    VideoPromptMessageContent,
                    AudioPromptMessageContent,
                    )
                ):
        if content.url:
            return types.Part.from_uri(
                file_uri=content.url,
                mime_type=content.mime_type,
            )
        else:
            return types.Part.from_bytes(
                data=base64.b64decode(content.base64_data),
                mime_type=content.mime_type,
            )
    else:
        type_tag = getattr(content, "type", None)
        # NOTE(QuantumGhost): The `PromptMessageContent` should be an ABC with an attribute named `type`. 
        # However, the current implementation does not include that attribute, so accessing `type` here will be flagged
        # by type checker.
        # 
        # Ignore the error for now.
        raise InvokeError(f"unknown content type, type={type_tag}, python_type={type(content)}")  # type: ignore
