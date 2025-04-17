from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import TypeVar

from pydantic_ai.agent import Agent, AgentRun, AgentRunResult
from pydantic_ai.mcp import MCPServer
from pydantic_ai.messages import (
    AgentStreamEvent,
    FinalResultEvent,
    HandleResponseEvent,
    ModelMessage,
    UserContent,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.result import ToolOutput
from pydantic_ai.tools import Tool
from pydantic_ai.usage import Usage

from lightblue_ai.log import logger
from lightblue_ai.mcps import get_mcp_servers
from lightblue_ai.models import infer_model
from lightblue_ai.prompts import get_system_prompt
from lightblue_ai.settings import Settings
from lightblue_ai.tools.manager import LightBlueToolManager
from lightblue_ai.utils import PendingMessage

OutputDataT = TypeVar("T")


class LightBlueAgent[OutputDataT]:
    def __init__(
        self,
        model: str | Model | None = None,
        system_prompt: str | None = None,
        result_type: type[OutputDataT] = str,
        result_tool_name: str = "final_result",
        result_tool_description: str | None = None,
        tools: list[Tool] | None = None,
        mcp_servers: list[MCPServer] | None = None,
        retries: int = 3,
        max_description_length: int | None = None,
        strict: bool | None = None,
    ):
        self.settings = Settings()
        model = model or self.settings.default_model
        tools = tools or []
        mcp_servers = mcp_servers or []

        if not model:
            raise ValueError("model or ENV `DEFAULT_MODEL` must be set")
        model_name = model.model_name if isinstance(model, Model) else model
        logger.info(f"Using model: {model_name}")

        self.enable_multi_turn = self.settings.enable_multi_turn
        if "anthropic" not in model_name and not isinstance(model, FunctionModel):
            max_description_length = max_description_length or 1000
            self.tool_return_data = False
            self.enable_multi_turn = self.settings.enable_multi_turn
        else:
            max_description_length = None
            self.tool_return_data = True
            # Disable multi-turn mode for anthropic model
            self.enable_multi_turn = False
        logger.info(
            f"Current multi-turn mode: {self.enable_multi_turn}, tool return data: {self.tool_return_data}, max description length: {max_description_length}"
        )

        self.tool_manager = LightBlueToolManager(max_description_length=max_description_length, strict=strict)
        self.agent = Agent[result_type](
            infer_model(model),
            output_type=(
                ToolOutput(
                    type_=result_type,
                    name=result_tool_name,
                    description=result_tool_description,
                    strict=strict,
                )
                if result_type is not str
                else str
            ),
            system_prompt=system_prompt or get_system_prompt(),
            tools=[*tools, *self.tool_manager.get_all_tools()],
            mcp_servers=[*mcp_servers, *get_mcp_servers()],
            retries=retries,
            deps_type=PendingMessage,
        )

    async def run(
        self,
        user_prompt: str | Sequence[UserContent],
        *,
        message_history: None | list[ModelMessage] = None,
        usage: None | Usage = None,
    ) -> AgentRunResult[OutputDataT]:
        messages = PendingMessage(multi_turn=self.enable_multi_turn, tool_return_data=self.tool_return_data)
        async with self.agent.run_mcp_servers():
            result = await self.agent.run(user_prompt, message_history=message_history, deps=messages)
            if usage:
                usage.incr(result.usage(), requests=1)

            while messages.has_messages():
                mess = messages.model_copy(deep=True)
                messages.clear()
                result = await self.agent.run(
                    ["File attachment", *mess.messages],
                    message_history=result.all_messages(),
                    usage=usage,
                    deps=messages,
                )
                if usage:
                    usage.incr(result.usage(), requests=1)

        return result

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[UserContent],
        *,
        message_history: None | list[ModelMessage] = None,
        usage: None | Usage = None,
    ) -> AsyncIterator[AgentRun]:
        async with (
            self.agent.run_mcp_servers(),
            self.agent.iter(
                user_prompt,
                message_history=message_history,
                deps=PendingMessage(
                    multi_turn=self.enable_multi_turn,
                    tool_return_data=self.tool_return_data,
                ),
            ) as run,
        ):
            yield run
        if usage:
            usage.incr(run.usage(), requests=1)

    async def iter_multiple(
        self,
        user_prompts: Sequence[str | Sequence[UserContent]],
        *,
        message_history: None | list[ModelMessage] = None,
        usage: None | Usage = None,
    ) -> AsyncIterator[AgentRun]:
        pending_messages = PendingMessage(multi_turn=self.enable_multi_turn, tool_return_data=self.tool_return_data)

        async with (
            self.agent.run_mcp_servers(),
        ):
            async with self.agent.iter(user_prompts, message_history=message_history, deps=pending_messages) as run:
                yield run

            if usage:
                usage.incr(run.usage(), requests=1)

            while pending_messages.has_messages():
                mess = pending_messages.model_copy(deep=True)
                pending_messages.clear()
                async with (
                    self.agent.iter(
                        ["File attachment", *mess.messages],
                        message_history=run.result.all_messages(),
                        usage=usage,
                        deps=pending_messages,
                    ) as run,
                ):
                    yield run

                if usage:
                    usage.incr(run.usage(), requests=1)

    async def yield_response_event(self, run: AgentRun) -> AsyncIterator[HandleResponseEvent | AgentStreamEvent]:
        """
        Yield the response event from the node.
        """
        async for node in run:
            if Agent.is_user_prompt_node(node) or Agent.is_end_node(node):
                continue

            elif Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        if not event or isinstance(event, FinalResultEvent):
                            continue
                        yield event
            else:
                logger.warning(f"Unknown node: {node}")
