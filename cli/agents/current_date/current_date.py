import asyncio
from typing import Annotated
from genai_session.session import GenAISession
from genai_session.utils.context import GenAIContext
from datetime import datetime

AGENT_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1YzBjZmNjOS0yNjgzLTRiZjAtYWI4Ny1kZDYxODk4ODAyNzYiLCJleHAiOjI1MzQwMjMwMDc5OSwidXNlcl9pZCI6IjdmOTgwYTg0LWE4N2ItNDVmMy05ODBkLTYxN2E0ZWY0NjI1OSJ9.MFokxHuCctgJzaIbG-osOdppmMnOfhFHXvh5QGYlfes" # noqa: E501
session = GenAISession(jwt_token=AGENT_JWT)


@session.bind(
    name="current_date",
    description="Agent that returns current date"
)
# async def current_date(
#     agent_context: GenAIContext,
#     test_arg: Annotated[
#         str,
#         "This is a test argument. Your agent can have as many parameters as you want. Feel free to rename or adjust it to your needs.",  # noqa: E501
#     ],
# ):
async def current_date(agent_context):
    agent_context.logger.info("Inside get_current_date")
    return datetime.now().strftime("%Y-%m-%d")


async def main():
    print(f"Agent with token '{AGENT_JWT}' started")
    await session.process_events()

if __name__ == "__main__":
    asyncio.run(main())
