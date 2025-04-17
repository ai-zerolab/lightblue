from pydantic import BaseModel  # pip install pydantic

from lightblue_ai.agent import LightBlueAgent  # pip install lightblue-ai

system_prompt = """Translate the given text from English to Chinese only.

You will be provided with a segment of text in English. Translate the text accurately into Chinese while preserving meaning, tone, and cultural nuances as much as possible.

# Steps

1. Confirm that the source language is English and the target language is Chinese.
2. Carefully read the input text to understand the context.
3. Translate the text into Chinese, maintaining the original meaning, tone, and style.
4. Adapt any idiomatic expressions appropriately to fit Chinese language and culture.
5. Review the translation for any grammatical or syntactical errors.

# Output Format

The translated text should be fully written in Chinese, formatted as a standalone paragraph or series of paragraphs, respecting the original structure.

# Examples

- **Input**: Translate "Hello! How are you?" from English to Chinese.
  - **Output**: "你好！你怎么样？"
- **Input**: Translate "Good morning, everyone." from English to Chinese.
  - **Output**: "大家早上好。"

# Notes

- Translator should handle common linguistic nuances and idioms.
- Provide translations that are culturally sensitive and appropriate for Chinese-speaking audiences.
"""

user_prompt_template = "<USER_PROMPT>{user_input}</USER_PROMPT>"


class TranslateResult(BaseModel):
    translated_text: str


async def main():
    user_input = "What is the meaning of life?"
    agent = LightBlueAgent(
        system_prompt=system_prompt,
        result_type=TranslateResult,
        result_tool_description="Return the translation of the input text.",
    )

    user_prompt = user_prompt_template.format(user_input=user_input)
    result = await agent.run(user_prompt)
    print(result.output.translated_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
