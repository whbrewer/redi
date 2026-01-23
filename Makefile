.PHONY: install install-dev test lint format install-skill install-skill-global

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=redi --cov-report=term-missing

# Linting & Formatting
lint:
	ruff check redi/ tests/

format:
	black redi/ tests/

# Skill installation for AI assistants
install-skill:
	@./scripts/install-skill.sh --local all

install-skill-global:
	@./scripts/install-skill.sh --global all

install-skill-claude:
	@./scripts/install-skill.sh claude

install-skill-gemini:
	@./scripts/install-skill.sh gemini

install-skill-codex:
	@./scripts/install-skill.sh codex

help:
	@echo "REDI Makefile targets:"
	@echo "  install              Install package in editable mode"
	@echo "  install-dev          Install with dev dependencies"
	@echo "  test                 Run tests"
	@echo "  test-cov             Run tests with coverage"
	@echo "  lint                 Run ruff linter"
	@echo "  format               Format code with black"
	@echo "  install-skill        Install skill for all AI assistants (local)"
	@echo "  install-skill-global Install skill globally for all AI assistants"
	@echo "  install-skill-claude Install skill for Claude Code only"
	@echo "  install-skill-gemini Install skill for Gemini CLI only"
	@echo "  install-skill-codex  Install skill for Codex only"
