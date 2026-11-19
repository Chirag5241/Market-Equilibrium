"""Check that every Azure OpenAI deployment in .env answers a chat request and a tool call.

Run:  uv run --with openai --with python-dotenv agents/check_azure.py
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))

client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"],  # .../openai/v1 (Azure's OpenAI-compatible v1 API)
    api_key=os.environ["AZURE_OPENAI_KEY"],
)

deployments = [v for k, v in sorted(os.environ.items()) if k.startswith("DEPLOYMENT_NAME_")]

# One tool, shaped like the negotiators' calculator, to confirm function calling works.
TOOLS = [{
    "type": "function",
    "name": "utility_change",
    "description": "Return the change in an investor's bond utility for a proposed trade.",
    "parameters": {
        "type": "object",
        "properties": {"delta": {"type": "array", "items": {"type": "number"}}},
        "required": ["delta"],
    },
}]

for name in deployments:
    row = {"deployment": name}
    try:
        t = time.time()
        r = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_completion_tokens=200,
        )
        row["chat"] = (r.choices[0].message.content or "").strip()[:40]
        row["model"] = r.model
        row["secs"] = round(time.time() - t, 1)

        # Tool calls go through the Responses API: the gpt-6 deployments reject
        # function tools on /chat/completions when reasoning is on.
        r = client.responses.create(
            model=name,
            input="Use the tool to evaluate buying 1 unit of bond 0 out of 6 bonds.",
            tools=TOOLS,
            tool_choice="required",
            max_output_tokens=1000,
        )
        calls = [o for o in r.output if o.type == "function_call"]
        row["tool_call"] = f"{calls[0].name}({calls[0].arguments})" if calls else "none"
        row["tokens"] = r.usage.total_tokens
    except Exception as e:  # report and keep going
        row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    print(json.dumps(row))
