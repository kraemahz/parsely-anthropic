import logging
import json
import time

from collections import deque
from dataclasses import is_dataclass

import anthropic
from anthropic.types import TextBlock
from anthropic.types.beta.tools import ToolUseBlock
from parsely_tools import to_dict, try_json_load

_logger = logging.getLogger(__name__)


def transform_tools(tools):
    dict_tools = []
    for tool in tools:
        if is_dataclass(tool):
            tool = to_dict(tool)
        elif isinstance(tool, dict):
            pass
        else:
            raise TypeError(f"Invalid tool type: {tool}")

        if 'parameters' in tool:
            tool['input_schema'] = tool.pop('parameters')
        dict_tools.append(tool)

    return dict_tools


CLAUDE_OPUS = "claude-3-opus-20240229"
MAX_RETRIES = 3


class ClaudeChat:
    def __init__(self,
                 model,
                 system_prompt,
                 *,
                 tools=None,
                 tool_provider=None):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_provider = tool_provider
        self._messages = []
        self._max_tokens = 4096
        self._client = anthropic.Anthropic()

    def add_message(self, role, content):
        self._messages.append({"role": role, "content": content})

    def _text_completion(self, max_retries: int = MAX_RETRIES):
        for i in range(max_retries):
            try:
                return self._client.beta.tools.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    max_tokens=self._max_tokens,
                    tools=self.tools,
                    messages=self._messages
                )
            except Exception:
                time.sleep(.2)
        raise RuntimeError("Failed to get response after multiple retries")

    def reset(self):
        self._messages = []

    def get_response(self, query, stop_on_tool=False, reset=True):
        self.add_message('user', query)
        response = self._text_completion()

        if len(response.content) > 1:
            content = response.content[-1]
        else:
            content = response.content[0]
        responses = deque([content])

        while responses:
            content = responses.popleft()
            if isinstance(content, TextBlock):
                self.add_message('assistant', content.text)
            else:
                tool_id = content.id
                self.add_message(
                    'assistant',
                    [{
                        "type": "tool_use",
                        "id": tool_id,
                        "name": content.name,
                        "input": content.input
                    }]
                )
                response = self.handle_tool_call(content)
                if stop_on_tool:
                    self.reset()
                    return response
                self.add_message(
                    "user",
                    [{
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": json.dumps(response)
                    }]
                )
                response = self._text_completion()
                responses.extend(response.content)

        if reset:
            self.reset()
        return content.text

    def handle_tool_call(self, tool: ToolUseBlock):
        """ToolUseBlock(id='toolu_01GhD5mx3AHeQbY9Mm4VPb7h',
                        input={'location': 'San Francisco, CA'},
                        name='get_weather', type='tool_use')"""
        tool_name = tool.name
        tool_input = tool.input
        _logger.info("Tool call: %s(%s)", tool_name, tool_input)
        return self.tool_provider(tool_name, tool_input)


class ClaudeChatTool:
    TOOLS = None
    TOOL_PROVIDER = None

    def __init__(self, **tool_provider_kwargs):
        tool_provider = (
            None
            if self.TOOL_PROVIDER is None
            else self.TOOL_PROVIDER(**tool_provider_kwargs)
        )

        if self.TOOLS:
            dict_tools = transform_tools(self.TOOLS)
        else:
            dict_tools = None

        self.client = ClaudeChat(
            self.MODEL,
            self.PROMPT,
            format=self.RESPONSE_FORMAT,
            tools=dict_tools,
            tool_provider=tool_provider,
        )

    def __call__(self, query):
        result = self.client.get_response(query,
                                          clear=True,
                                          upgrade=self.UPGRADE)
        if self.RESPONSE_FORMAT == {"type": "json_object"}:
            return try_json_load(result)
        return result
