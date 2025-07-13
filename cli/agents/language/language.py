import asyncio
from typing import Annotated
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3N2RiOWYwNC1jZDQ4LTQzNTktYjJiZi0zNmRmOTU2YzBlZWEiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6IjdmOTgwYTg0LWE4N2ItNDVmMy05ODBkLTYxN2E0ZWY0NjI1OSJ9.8agATd5E0oRBxm8npr0qvH-ynMA59T4mXOA7M6VuBG8" # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)


@session.bind(
    name="language",
    description="Agent that returns different language"
)
async def language(
    agent_context: GenAIContext,
    test_arg: Annotated[
        str,
        "This is a test argument. Your agent can have as many parameters as you want. Feel free to rename or adjust it to your needs.",  # noqa: E501
    ],
):
    """Agent that returns different language"""
    return "Hello, World!"


async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
