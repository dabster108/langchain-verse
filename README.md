# langchain-verse

LangChain + Mistral AI framework for intelligent agents

## Setup

1. **Install UV:**

```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
```

2. **Install dependencies:**

```bash
   uv sync
```

3. **Setup environment:**

```bash
   cp .env.example .env
   # Edit .env and add your MISTRAL_API_KEY
```

4. **Run:**

```bash
   uv run python -m langchain_verse.main
```

## Environment Variables

- `MISTRAL_API_KEY` - Your Mistral AI API key (required)

See `.env.example` for template.

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black src/

# Lint
uv run ruff check src/
```
