from jinja2 import Environment, FileSystemLoader

from lightblue_ai.agent import LightBlueAgent

env = Environment(loader=FileSystemLoader("."), autoescape=True)

template = env.get_template("prompt-template.txt")


async def main():
    agent = LightBlueAgent(model="bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    objective = input("What website would you like made?: ")
    prompt = template.render(objective=objective)

    result = await agent.run(prompt)
    print(result.output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
