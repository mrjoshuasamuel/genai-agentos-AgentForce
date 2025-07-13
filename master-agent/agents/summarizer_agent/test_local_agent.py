import asyncio
from summarizer_agent.agent import text_summarizer

async def run_test():
    result = await text_summarizer(
        agent_context=None,
        text="""Artificial Intelligence is transforming industries by automating tasks,
                improving efficiency, and creating new business opportunities.
                It enables predictive analytics, intelligent automation, and better decision-making.""",
        summary_type="extractive",
        max_length=50,
        key_points=2
    )

    print("===== Summary Result =====")
    print("Success:", result.get("success"))
    if result.get("success"):
        print("Summary Type:", result["data"].get("summary_type"))
        print("Summary:", result["data"].get("summary"))
        print("Words:", result["data"].get("length"))
    else:
        print("Error Type:", result.get("error"))
        print("Message:", result.get("message"))

asyncio.run(run_test())
