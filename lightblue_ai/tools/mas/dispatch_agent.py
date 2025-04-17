from pathlib import Path
from typing import Annotated

from pydantic import Field
from pydantic_ai import Agent, BinaryContent

from lightblue_ai.log import logger
from lightblue_ai.models import infer_model
from lightblue_ai.settings import Settings
from lightblue_ai.tools.base import LightBlueTool, Scope
from lightblue_ai.tools.extensions import hookimpl
from lightblue_ai.tools.manager import LightBlueToolManager
from lightblue_ai.tools.media_mixin import MediaMixin


class DispatchAgentTool(LightBlueTool, MediaMixin):
    def __init__(self, manager: LightBlueToolManager):
        self.name = "context_agent"
        self.settings = Settings()
        self.scopes = [Scope.exec]
        self.manager = manager
        self.description = """Launch a new agent that has access to the following tools: GlobTool, GrepTool, LS, View and others for searching information.

When you are searching for a keyword or file and are not confident that you will find the right match on the first try, use this tool to perform the search for you. For example:

- If you are searching for a keyword like "config" or "logger", this tool is appropriate.
- If you want to read a specific file path, use the View or GlobTool tool instead to find the match more quickly.
- If you are searching for a specific class definition like "class Foo", use the GlobTool tool instead to find the match more quickly.

Usage notes:

1. Launch multiple agents concurrently whenever possible to maximize performance; to do that, use a single message with multiple tool uses.
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously, and you should specify exactly what information the agent should return in its final and only message to you.
4. The agent's outputs should generally be trusted.
5. IMPORTANT: The agent cannot use Bash, Replace, Edit, so it cannot modify files. If you need to use these tools, use them directly instead of going through the agent.
"""

    async def call(
        self,
        system_prompt: Annotated[str, Field(description="System prompt for the agent.")],
        objective: Annotated[str, Field(description="The objective to achieve.")],
        attatchments: Annotated[
            list[str] | None,
            Field(
                default=None,
                description="A list of file paths to attach to the agent.",
            ),
        ] = None,
    ) -> str:
        model_name = self.settings.sub_agent_model or self.settings.default_model
        max_description_length = 1000 if "anthropic" in model_name else None

        self.agent = Agent(
            infer_model(model_name),
            system_prompt=system_prompt,
            tools=self.manager.get_sub_agent_tools(max_description_length=max_description_length),
        )

        attatchments = [
            Path(a).expanduser().resolve().absolute()
            for a in attatchments or []
            if Path(a).exists() and Path(a).is_file()
        ]

        attatchment_data = []
        for path in attatchments:
            if path.suffix.lower() in self.binary_extensions:
                # Return binary content for binary files
                with path.open("rb") as f:
                    content = f.read()
                media_type = self._get_mime_type(path)
                if media_type.startswith("image/"):
                    content = self._resized_image(content)
                data = BinaryContent(data=content, media_type=self._get_mime_type(path))
                attatchment_data.append(data)
                logger.info(f"{path} attatchment added")
        return (await self.agent.run([*attatchment_data, objective])).output


class ReflactionAgentTool(DispatchAgentTool):
    def __init__(self, manager: LightBlueToolManager):
        self.name = "reflaction_agent"
        self.settings = Settings()
        self.scopes = [Scope.exec]
        self.manager = manager
        self.description = """Launch a reflection agent that evaluates completed tasks and provides improvement feedback.

When you have completed a task and want to verify its correctness, quality, or identify potential improvements, use this tool to perform an objective assessment. The reflection agent will:

- Analyze the completed task against the original requirements
- Identify any errors, omissions, or potential issues
- Evaluate the quality and effectiveness of the solution
- Suggest specific improvements or alternative approaches
- Provide a confidence score regarding the correctness of the solution

Usage notes:

1. Provide the reflection agent with: (a) the original task requirements, (b) the completed solution, and (c) any specific evaluation criteria you want addressed.
2. The agent will return a single comprehensive evaluation message containing its analysis and recommendations.
3. The evaluation result is not visible to the user automatically. To share insights with the user, you should send a text message summarizing the key findings.
4. Each reflection agent invocation is stateless. Your prompt should contain all the context needed for a thorough evaluation, including the complete task description and solution.
5. The reflection agent excels at identifying logical errors, edge cases, optimizations, and alignment with requirements that might have been overlooked during initial implementation.
6. If multiple evaluation perspectives are needed, launch multiple reflection agents concurrently with different evaluation criteria.
7. The reflection agent can evaluate code, writing, plans, decisions, and other outputs, but cannot execute code or make changes to files.
8. For maximum value, include specific questions or concerns you want the reflection agent to address in its evaluation.
"""


@hookimpl
def register(manager):
    manager.register(DispatchAgentTool(manager))
    manager.register(ReflactionAgentTool(manager))
