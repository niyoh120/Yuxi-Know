from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.types import Command

import src.agents.common.middlewares.runtime_config_middleware as runtime_middleware
from src.agents.common.middlewares.runtime_config_middleware import RuntimeConfigMiddleware


@dataclass
class _FakeTool:
    name: str


@dataclass
class _FakeRequest:
    runtime: Any
    tools: list[Any]
    system_message: SystemMessage
    state: dict[str, Any]

    def override(self, **kwargs):
        return _FakeRequest(
            runtime=kwargs.get("runtime", self.runtime),
            tools=kwargs.get("tools", self.tools),
            system_message=kwargs.get("system_message", self.system_message),
            state=kwargs.get("state", self.state),
        )


@dataclass
class _FakeToolCallRequest:
    tool_call: dict[str, Any]
    runtime: Any
    state: dict[str, Any]


async def _echo_handler(request):
    return request


def _build_request(*, skills: list[str], tools: list[str], system_prompt: str = "你是助手", state=None) -> _FakeRequest:
    context = SimpleNamespace(system_prompt=system_prompt, skills=skills, tools=[], knowledges=[], mcps=[])
    runtime = SimpleNamespace(context=context)
    return _FakeRequest(
        runtime=runtime,
        tools=[_FakeTool(name=name) for name in tools],
        system_message=SystemMessage(content=[{"type": "text", "text": "base"}]),
        state=state or {},
    )


def _build_tool_request(*, skills: list[str], visible_skills: list[str], file_path: str) -> _FakeToolCallRequest:
    return _FakeToolCallRequest(
        tool_call={"name": "read_file", "args": {"file_path": file_path}},
        runtime=SimpleNamespace(context=SimpleNamespace(skills=skills)),
        state={
            "skill_session_snapshot": {
                "selected_skills": skills,
                "visible_skills": visible_skills,
                "prompt_metadata": {},
                "dependency_map": {},
            }
        },
    )


def _extract_appended_prompt(request: _FakeRequest) -> str:
    return request.system_message.content_blocks[-1]["text"]


def _build_middleware() -> RuntimeConfigMiddleware:
    return RuntimeConfigMiddleware(
        enable_model_override=False,
        enable_tools_override=False,
        enable_system_prompt_override=True,
        enable_skills_prompt_override=True,
    )


def _build_snapshot(selected: list[str], metadata: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "selected_skills": selected,
        "visible_skills": selected,
        "prompt_metadata": metadata or {},
        "dependency_map": {},
    }


@pytest.mark.asyncio
async def test_injects_skills_section_when_skills_configured_and_read_file_available(monkeypatch: pytest.MonkeyPatch):
    async def fake_resolve(selected):
        assert selected == ["research-report"]
        return _build_snapshot(
            ["research-report"],
            {
                "research-report": {
                    "name": "research-report",
                    "description": "Write structured research reports",
                    "path": "/skills/research-report/SKILL.md",
                }
            },
        )

    monkeypatch.setattr(runtime_middleware, "resolve_session_snapshot", fake_resolve)
    middleware = _build_middleware()
    request = _build_request(skills=["research-report"], tools=["read_file"])

    result = await middleware.awrap_model_call(request, _echo_handler)
    prompt = _extract_appended_prompt(result)

    assert "## Skills System" in prompt
    assert "**Skills Skills**: `/skills/` (higher priority)" in prompt
    assert "- **research-report**: Write structured research reports" in prompt
    assert "Read `/skills/research-report/SKILL.md` for full instructions" in prompt
    assert "Recognize when a skill applies" in prompt
    assert "当前时间：" in prompt
    assert "skill_session_snapshot" in result.state


@pytest.mark.asyncio
async def test_skips_skills_section_when_context_skills_empty():
    middleware = _build_middleware()
    request = _build_request(skills=[], tools=["read_file"])

    result = await middleware.awrap_model_call(request, _echo_handler)
    prompt = _extract_appended_prompt(result)

    assert "## Skills System" not in prompt


@pytest.mark.asyncio
async def test_skips_skills_section_without_read_file_and_logs_warning(monkeypatch: pytest.MonkeyPatch):
    warnings: list[str] = []
    fake_logger = SimpleNamespace(
        debug=lambda *_args, **_kwargs: None,
        warning=lambda message: warnings.append(message),
    )
    monkeypatch.setattr(runtime_middleware, "logger", fake_logger)

    async def fake_resolve(_selected):
        return _build_snapshot(
            ["research-report"],
            {
                "research-report": {
                    "name": "research-report",
                    "description": "Write structured research reports",
                    "path": "/skills/research-report/SKILL.md",
                }
            },
        )

    monkeypatch.setattr(runtime_middleware, "resolve_session_snapshot", fake_resolve)
    middleware = _build_middleware()
    request = _build_request(skills=["research-report"], tools=["write_file"])

    result = await middleware.awrap_model_call(request, _echo_handler)
    prompt = _extract_appended_prompt(result)

    assert "## Skills System" not in prompt
    assert any("read_file unavailable" in msg for msg in warnings)


@pytest.mark.asyncio
async def test_injects_skills_in_input_order_with_dedup_and_invalid_slug_skipped(monkeypatch: pytest.MonkeyPatch):
    async def fake_resolve(_selected):
        return _build_snapshot(
            ["beta", "missing", "alpha", "beta"],
            {
                "beta": {"name": "beta", "description": "beta skill", "path": "/skills/beta/SKILL.md"},
                "alpha": {"name": "alpha", "description": "alpha skill", "path": "/skills/alpha/SKILL.md"},
            },
        )

    monkeypatch.setattr(runtime_middleware, "resolve_session_snapshot", fake_resolve)
    middleware = _build_middleware()
    request = _build_request(skills=["beta", "missing", "alpha", "beta"], tools=["read_file"])

    result = await middleware.awrap_model_call(request, _echo_handler)
    prompt = _extract_appended_prompt(result)

    beta_line = "- **beta**: beta skill"
    alpha_line = "- **alpha**: alpha skill"
    assert beta_line in prompt
    assert alpha_line in prompt
    assert prompt.find(beta_line) < prompt.find(alpha_line)
    assert prompt.count(beta_line) == 1
    assert "missing" not in prompt


@pytest.mark.asyncio
async def test_awrap_tool_call_activates_skill_when_read_skill_md():
    middleware = _build_middleware()
    request = _build_tool_request(
        skills=["research-report"],
        visible_skills=["research-report"],
        file_path="/skills/research-report/SKILL.md",
    )

    async def _handler(_request):
        return ToolMessage(content="ok", tool_call_id="tc-1")

    result = await middleware.awrap_tool_call(request, _handler)
    assert isinstance(result, Command)
    assert result.update["activated_skills"] == ["research-report"]
    assert len(result.update["messages"]) == 1


@pytest.mark.asyncio
async def test_awrap_tool_call_skips_invalid_skill_slug_path():
    middleware = _build_middleware()
    request = _build_tool_request(
        skills=["research-report"],
        visible_skills=["research-report"],
        file_path="/skills/../SKILL.md",
    )

    async def _handler(_request):
        return ToolMessage(content="ok", tool_call_id="tc-1")

    result = await middleware.awrap_tool_call(request, _handler)
    assert isinstance(result, ToolMessage)


@pytest.mark.asyncio
async def test_awrap_tool_call_merges_with_existing_command_update():
    middleware = _build_middleware()
    request = _build_tool_request(
        skills=["research-report"],
        visible_skills=["research-report"],
        file_path="/skills/research-report/SKILL.md",
    )

    async def _handler(_request):
        return Command(update={"messages": [ToolMessage(content="ok", tool_call_id="tc-1")], "activated_skills": ["a"]})

    result = await middleware.awrap_tool_call(request, _handler)
    assert isinstance(result, Command)
    assert result.update["activated_skills"] == ["a", "research-report"]


@pytest.mark.asyncio
async def test_awrap_tool_call_denies_invisible_skill():
    middleware = _build_middleware()
    request = _build_tool_request(
        skills=["research-report"],
        visible_skills=["alpha"],
        file_path="/skills/research-report/SKILL.md",
    )

    async def _handler(_request):
        return ToolMessage(content="ok", tool_call_id="tc-1")

    result = await middleware.awrap_tool_call(request, _handler)
    assert isinstance(result, ToolMessage)


@pytest.mark.asyncio
async def test_model_call_injects_dependency_tools_and_mcps_after_activation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        runtime_middleware,
        "get_buildin_tools",
        lambda: [_FakeTool(name="calculator"), _FakeTool(name="dep-tool")],
    )
    monkeypatch.setattr(runtime_middleware, "get_kb_based_tools", lambda db_names=None: [])
    monkeypatch.setattr(
        runtime_middleware,
        "build_dependency_bundle",
        lambda _snapshot, activated: {"tools": ["dep-tool"], "mcps": ["mcp-a"], "skills": activated},
    )

    async def fake_get_enabled_mcp_tools(server_name: str):
        if server_name == "mcp-a":
            return [_FakeTool(name="mcp_tool")]
        return []

    monkeypatch.setattr(runtime_middleware, "get_enabled_mcp_tools", fake_get_enabled_mcp_tools)

    middleware = RuntimeConfigMiddleware(
        extra_tools=[_FakeTool(name="mcp_tool")],
        enable_model_override=False,
        enable_tools_override=True,
        enable_system_prompt_override=False,
        enable_skills_prompt_override=False,
    )

    request = _build_request(
        skills=[],
        tools=["calculator", "dep-tool", "mcp_tool", "read_file"],
        state={"activated_skills": ["alpha"], "skill_session_snapshot": _build_snapshot([])},
    )

    result = await middleware.awrap_model_call(request, _echo_handler)
    tool_names = [t.name for t in result.tools]
    assert "dep-tool" in tool_names
    assert "mcp_tool" in tool_names
    assert "calculator" not in tool_names


@pytest.mark.asyncio
async def test_model_call_reuses_snapshot_until_skills_changed(monkeypatch: pytest.MonkeyPatch):
    called = {"count": 0}

    async def fake_resolve(selected):
        called["count"] += 1
        return _build_snapshot(selected)

    monkeypatch.setattr(runtime_middleware, "resolve_session_snapshot", fake_resolve)

    middleware = _build_middleware()
    req1 = _build_request(skills=["alpha"], tools=["read_file"], state={})
    res1 = await middleware.awrap_model_call(req1, _echo_handler)
    assert called["count"] == 1

    req2 = _build_request(skills=["alpha"], tools=["read_file"], state=res1.state)
    await middleware.awrap_model_call(req2, _echo_handler)
    assert called["count"] == 1

    req3 = _build_request(skills=["beta"], tools=["read_file"], state=res1.state)
    await middleware.awrap_model_call(req3, _echo_handler)
    assert called["count"] == 2


@pytest.mark.asyncio
async def test_injects_dependency_skills_into_prompt(monkeypatch: pytest.MonkeyPatch):
    async def fake_resolve(_selected):
        return {
            "selected_skills": ["alpha"],
            "visible_skills": ["alpha", "beta"],
            "prompt_metadata": {
                "alpha": {"name": "alpha", "description": "alpha desc", "path": "/skills/alpha/SKILL.md"},
                "beta": {"name": "beta", "description": "beta desc", "path": "/skills/beta/SKILL.md"},
            },
            "dependency_map": {
                "alpha": {"tools": [], "mcps": [], "skills": ["beta"]},
                "beta": {"tools": [], "mcps": [], "skills": []},
            },
        }

    monkeypatch.setattr(runtime_middleware, "resolve_session_snapshot", fake_resolve)
    middleware = _build_middleware()
    request = _build_request(skills=["alpha"], tools=["read_file"])

    result = await middleware.awrap_model_call(request, _echo_handler)
    prompt = _extract_appended_prompt(result)
    assert "- **alpha**: alpha desc" in prompt
    assert "- **beta**: beta desc" in prompt
