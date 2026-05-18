"""Simple local agent with workspace file tools."""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mistralai import ChatMistralAI

from .config import AppConfig, load_config


def build_model(config: AppConfig) -> ChatMistralAI:
    return ChatMistralAI(
        model=config.model_name,
        temperature=0,
    )


def build_tools(workspace_root: Path):
    @tool
    def list_files(relative_path: str = ".") -> str:
        """List files and folders inside the workspace."""

        target = (workspace_root / relative_path).resolve()
        if workspace_root not in target.parents and target != workspace_root:
            return "Path escapes the workspace root."
        if not target.exists():
            return f"Path does not exist: {relative_path}"
        if target.is_file():
            return str(target.relative_to(workspace_root))

        entries = sorted(item.name + ("/" if item.is_dir() else "") for item in target.iterdir())
        if not entries:
            return f"{relative_path}: (empty)"
        return "\n".join(entries)

    @tool
    def read_file(relative_path: str) -> str:
        """Read a text file from the workspace."""

        target = (workspace_root / relative_path).resolve()
        if workspace_root not in target.parents and target != workspace_root:
            return "Path escapes the workspace root."
        if not target.exists() or not target.is_file():
            return f"File does not exist: {relative_path}"

        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "The file is not plain UTF-8 text."

    return [list_files, read_file]


def run_agent(question: str, config: AppConfig | None = None) -> str:
    config = config or load_config()
    tools = build_tools(config.workspace_root)
    tool_map = {agent_tool.name: agent_tool for agent_tool in tools}
    model = build_model(config).bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a small assistant for this workspace. "
                "Use the available tools to inspect files when needed, then answer clearly. "
                f"Workspace root: {config.workspace_root}"
            )
        ),
        HumanMessage(content=question),
    ]

    for _ in range(config.max_iterations):
        response = model.invoke(messages)
        messages.append(response)

        if not isinstance(response, AIMessage) or not response.tool_calls:
            return response.content if isinstance(response.content, str) else str(response.content)

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_result = tool_map[tool_name].invoke(tool_args)
            messages.append(
                ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
            )

    raise RuntimeError("Agent exceeded the maximum number of tool iterations.")


def interactive_main() -> None:
    config = load_config()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:]).strip()
        print(run_agent(question, config))
        return

    print("Simple workspace agent ready. Type a question or 'exit'.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print(run_agent(question, config))


def main() -> None:
    interactive_main()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()