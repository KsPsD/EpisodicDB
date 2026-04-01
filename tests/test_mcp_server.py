import json
import random
import sys

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


@pytest.fixture
def server_params(tmp_path):
    db_path = str(tmp_path / "test.db")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "episodicdb.mcp", "--agent-id", "test", "--db", db_path],
    )


# --- list tools ---

@pytest.mark.asyncio
async def test_list_tools(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            assert len(tool_names) == 9
            assert "record_episode" in tool_names
            assert "record_tool_call" in tool_names
            assert "record_decision" in tool_names
            assert "top_failing_tools" in tool_names
            assert "similar_episodes" in tool_names


# --- writer tools ---

@pytest.mark.asyncio
async def test_record_episode_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("record_episode", {"status": "success"})
            text = result.content[0].text
            assert len(text) == 36


@pytest.mark.asyncio
async def test_record_tool_call_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ep = await session.call_tool("record_episode", {"status": "success"})
            ep_id = ep.content[0].text
            result = await session.call_tool("record_tool_call", {
                "episode_id": ep_id,
                "tool_name": "Edit",
                "outcome": "failure",
                "error_message": "permission denied",
            })
            text = result.content[0].text
            assert len(text) == 36


@pytest.mark.asyncio
async def test_record_decision_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ep = await session.call_tool("record_episode", {"status": "success"})
            ep_id = ep.content[0].text
            result = await session.call_tool("record_decision", {
                "episode_id": ep_id,
                "rationale": "Read before modify",
                "decision_type": "safety_check",
            })
            text = result.content[0].text
            assert len(text) == 36


@pytest.mark.asyncio
async def test_record_episode_invalid_status(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("record_episode", {"status": "invalid"})
            assert result.isError is True


@pytest.mark.asyncio
async def test_agent_id_override(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("record_episode", {
                "status": "success",
                "agent_id": "custom-agent",
            })
            text = result.content[0].text
            assert len(text) == 36


# --- analytics tools ---

@pytest.mark.asyncio
async def test_top_failing_tools_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ep = await session.call_tool("record_episode", {"status": "failure"})
            ep_id = ep.content[0].text
            await session.call_tool("record_tool_call", {
                "episode_id": ep_id, "tool_name": "Edit", "outcome": "failure",
            })
            result = await session.call_tool("top_failing_tools", {"days": 30})
            data = json.loads(result.content[0].text)
            assert isinstance(data, list)


@pytest.mark.asyncio
async def test_never_succeeded_tools_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            ep = await session.call_tool("record_episode", {"status": "failure"})
            ep_id = ep.content[0].text
            await session.call_tool("record_tool_call", {
                "episode_id": ep_id, "tool_name": "NeverWorks", "outcome": "failure",
            })
            result = await session.call_tool("never_succeeded_tools", {})
            data = json.loads(result.content[0].text)
            assert isinstance(data, list)


@pytest.mark.asyncio
async def test_hourly_failure_rate_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("hourly_failure_rate", {"days": 30})
            data = json.loads(result.content[0].text)
            assert isinstance(data, list)


@pytest.mark.asyncio
async def test_compare_periods_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("compare_periods", {
                "metric": "failure_rate", "days": 7,
            })
            data = json.loads(result.content[0].text)
            assert "period_a" in data
            assert "period_b" in data
            assert "delta" in data


@pytest.mark.asyncio
async def test_before_failure_sequence_via_mcp(server_params):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("before_failure_sequence", {
                "tool_name": "Edit",
            })
            data = json.loads(result.content[0].text)
            assert isinstance(data, list)


@pytest.mark.asyncio
async def test_similar_episodes_via_mcp(server_params):
    random.seed(0)
    emb = [random.uniform(-1, 1) for _ in range(1536)]
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("record_episode", {
                "status": "success", "embedding": emb,
            })
            result = await session.call_tool("similar_episodes", {
                "embedding": emb, "limit": 3,
            })
            data = json.loads(result.content[0].text)
            assert isinstance(data, list)
