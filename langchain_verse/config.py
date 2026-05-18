"""Configuration helpers for the local LangChain agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
	model_name: str
	workspace_root: Path
	max_iterations: int


def load_config() -> AppConfig:
	load_dotenv()

	workspace_root = Path(__file__).resolve().parents[1]

	return AppConfig(
		model_name=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
		workspace_root=workspace_root,
		max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "5")),
	)
