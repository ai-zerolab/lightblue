import asyncio
import functools
import json
from pathlib import Path

import typer
from pydantic_ai import BinaryContent
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    HandleResponseEvent,
    ModelMessagesTypeAdapter,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ToolReturnPart,
)
from pydantic_ai.usage import Usage
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from lightblue_ai.agent import LightBlueAgent
from lightblue_ai.log import logger
from lightblue_ai.mcps import get_mcp_servers

app = typer.Typer()


def make_sync(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class ResponseEventHandler:
    def __init__(self):
        self.events = []  # Track all events in chronological order
        self.parts = {}  # Track parts by index for delta updates
        self.max_content_length = 1000  # Fixed truncation length

    def truncate_content(self, content: str) -> str:
        """Truncate content if it exceeds the maximum length."""
        if len(content) > self.max_content_length:
            return content[: self.max_content_length] + "... [truncated]"
        return content

    def update_from_event(self, event: HandleResponseEvent | AgentStreamEvent) -> str:
        # Handle different event types and update the display content
        if isinstance(event, PartStartEvent):
            self.parts[event.index] = event.part
            self.events.append(("text", event.part))
        elif isinstance(event, PartDeltaEvent):
            if event.index not in self.parts:
                logger.warning(f"Part index {event.index} not found in parts.")
                return self.format_content()
            # Apply delta to existing part
            if isinstance(event.delta, TextPartDelta):
                part = self.parts[event.index]
                if isinstance(part, TextPart):
                    part.content += event.delta.content_delta

                    # Update the corresponding event in the events list
                    for _, (event_type, event_data) in enumerate(self.events):
                        if event_type == "text" and event_data is part:
                            # No need to update since we're modifying the object directly
                            break
        elif isinstance(event, FunctionToolCallEvent):
            self.events.append(("tool_call", event.part))
        elif isinstance(event, FunctionToolResultEvent):
            self.events.append(("tool_result", event.result))

        # Generate formatted content as Markdown
        return self.format_content()

    def format_content(self) -> str:
        # Format the content for display as Markdown
        formatted = []

        # Process events in chronological order
        for event_type, event_data in self.events:
            if event_type == "text" and isinstance(event_data, TextPart):
                formatted.append(event_data.content)
            elif event_type == "tool_call":
                tool_call = event_data
                formatted.append(f"\n**Tool Call:** `{tool_call.tool_name}`\n")
                formatted.append("```json")

                # Truncate tool call arguments if necessary
                if isinstance(tool_call.args, dict):
                    args_str = json.dumps(tool_call.args, indent=2)
                    formatted.append(self.truncate_content(args_str))
                else:
                    formatted.append(self.truncate_content(str(tool_call.args)))

                formatted.append("```")
            elif event_type == "tool_result":
                result = event_data
                if isinstance(result, ToolReturnPart):
                    formatted.append("\n**Tool Result:**\n")
                    formatted.append("```")
                    if isinstance(result.content, BinaryContent):
                        formatted.append(
                            self.truncate_content(
                                f"Binary content({result.content.media_type}): {len(result.content.data)} bytes"
                            )
                        )
                    # Truncate tool result if necessary
                    else:
                        formatted.append(self.truncate_content(result.model_response_str()))
                    formatted.append("```")
                elif isinstance(result, RetryPromptPart):
                    formatted.append("\n**Tool Error:**\n")
                    formatted.append("```")
                    # Truncate error message if necessary
                    formatted.append(self.truncate_content(result.model_response()))
                    formatted.append("```")

        return "\n".join(formatted)


@app.command()
@make_sync
async def submit(
    prompt: str = typer.Argument("prompt.md", help="The prompt to send to the agent, text or file"),
    message_history_json: Path = typer.Option(
        default="message_history.json",
        help="The path to store the result",
    ),
    all_messages_json: Path = typer.Option(
        default="all_messages.json",
        help="The path to store the result",
    ),
):
    if Path(prompt).exists():
        with open(prompt) as f:
            prompt = f.read()

    message_history = None
    if Path(message_history_json).exists():
        message_history = ModelMessagesTypeAdapter.validate_json(message_history_json.read_bytes())

    agent = LightBlueAgent()
    usage = Usage()

    console = Console()

    console.print(Markdown(prompt))
    with console.status("[bold blue]Processing...[/bold blue]"):
        result = await agent.run(prompt, message_history=message_history, usage=usage)

    console.print(Markdown(result.output))

    with all_messages_json.open("wb") as f:
        f.write(result.all_messages_json())

    console.print(f"[bold green]Usage:[/bold green] {usage}")
    console.print(f"[bold green]Saved all messages to[/bold green] {all_messages_json.absolute().as_posix()}")


@app.command()
@make_sync
async def stream(
    prompt: str = typer.Argument("prompt.md", help="The prompt to send to the agent, text or file"),
    message_history_json: Path = typer.Option(
        default="message_history.json",
        help="The path to store the result",
    ),
    all_messages_json: Path = typer.Option(
        default="all_messages.json",
        help="The path to store the result",
    ),
):
    if Path(prompt).exists():
        with open(prompt) as f:
            prompt = f.read()

    message_history = None
    if Path(message_history_json).exists():
        message_history = ModelMessagesTypeAdapter.validate_json(message_history_json.read_bytes())

    usage = Usage()
    agent = LightBlueAgent()
    console = Console()
    event_handler = ResponseEventHandler()

    console.print(Markdown(prompt))
    with Live("", console=console, vertical_overflow="visible", refresh_per_second=1) as live:
        if agent.enable_multi_turn:
            async for run in agent.iter_multiple(prompt, message_history=message_history, usage=usage):
                async for event in agent.yield_response_event(run):
                    # Log the raw event for debugging
                    logger.debug(f"Event: {event}")
                    # Update the display with the new event
                    content = event_handler.update_from_event(event)
                    # Use Markdown for rendering
                    live.update(Markdown(content))
        else:
            async with agent.iter(prompt, message_history=message_history, usage=usage) as run:
                async for event in agent.yield_response_event(run):
                    # Log the raw event for debugging
                    logger.debug(f"Event: {event}")
                    # Update the display with the new event
                    content = event_handler.update_from_event(event)
                    # Use Markdown for rendering
                    live.update(Markdown(content))

        with all_messages_json.open("wb") as f:
            console.print(f"Saved current round to {all_messages_json.absolute().as_posix()}")
            f.write(run.result.all_messages_json())
        console.print(f"[bold green]Usage:[/bold green] {usage}")

    console.print(f"[bold green]All Usage:[/bold green] {usage}")


@app.command()
def status():
    agent = LightBlueAgent()

    logger.info(f"Found {len(agent.tool_manager.get_all_tools())} tools.")
    logger.info(f"Found {len(get_mcp_servers())} MCP servers.")
