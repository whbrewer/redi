#!/usr/bin/env bash
# Install REDI skill for AI coding assistants (Claude Code, Gemini CLI, Codex)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDI_ROOT="$(dirname "$SCRIPT_DIR")"
SKILL_FILE="$REDI_ROOT/SKILL.md"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] [TARGETS...]

Install REDI skill for AI coding assistants.

Targets:
  claude    Install for Claude Code
  gemini    Install for Gemini CLI
  codex     Install for OpenAI Codex CLI
  all       Install for all supported tools (default)

Options:
  --global  Install globally (~/.claude/, ~/.gemini/, ~/.codex/)
  --local   Install to current project directory (default)
  --link    Use symlinks instead of copying
  --help    Show this help message

Examples:
  $(basename "$0")                    # Install locally for all tools
  $(basename "$0") --global           # Install globally for all tools
  $(basename "$0") claude gemini      # Install locally for Claude and Gemini
  $(basename "$0") --global --link    # Install globally using symlinks
EOF
}

# Parse arguments
GLOBAL=false
USE_LINK=false
TARGETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --global) GLOBAL=true; shift ;;
        --local) GLOBAL=false; shift ;;
        --link) USE_LINK=true; shift ;;
        --help|-h) usage; exit 0 ;;
        claude|gemini|codex|all) TARGETS+=("$1"); shift ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# Default to all targets
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("all")
[[ " ${TARGETS[*]} " =~ " all " ]] && TARGETS=("claude" "gemini" "codex")

# Helper to install file
install_file() {
    local src="$1"
    local dst="$2"
    local name="$3"

    mkdir -p "$(dirname "$dst")"

    if $USE_LINK; then
        ln -sf "$src" "$dst"
        echo "  Linked $name -> $dst"
    else
        cp "$src" "$dst"
        echo "  Copied $name -> $dst"
    fi
}

# Install for Claude Code
install_claude() {
    echo "Installing for Claude Code..."
    if $GLOBAL; then
        # Global: add to ~/.claude/commands/ as a custom command
        install_file "$SKILL_FILE" "$HOME/.claude/commands/redi.md" "SKILL.md"
    else
        # Local: create CLAUDE.md in project root (append if exists)
        local dst="$PWD/CLAUDE.md"
        if [[ -f "$dst" ]]; then
            if ! grep -q "# REDI - Readiness Engine" "$dst" 2>/dev/null; then
                echo "" >> "$dst"
                cat "$SKILL_FILE" >> "$dst"
                echo "  Appended SKILL.md -> $dst"
            else
                echo "  REDI skill already in $dst"
            fi
        else
            install_file "$SKILL_FILE" "$dst" "SKILL.md"
        fi
    fi
}

# Install for Gemini CLI
install_gemini() {
    echo "Installing for Gemini CLI..."
    if $GLOBAL; then
        # Global: append to ~/.gemini/GEMINI.md
        local dst="$HOME/.gemini/GEMINI.md"
        mkdir -p "$HOME/.gemini"
        if [[ -f "$dst" ]]; then
            if ! grep -q "# REDI - Readiness Engine" "$dst" 2>/dev/null; then
                echo "" >> "$dst"
                # Strip YAML frontmatter for Gemini
                tail -n +5 "$SKILL_FILE" >> "$dst"
                echo "  Appended SKILL.md -> $dst"
            else
                echo "  REDI skill already in $dst"
            fi
        else
            tail -n +5 "$SKILL_FILE" > "$dst"
            echo "  Created $dst"
        fi
    else
        # Local: create GEMINI.md in project root
        local dst="$PWD/GEMINI.md"
        if [[ -f "$dst" ]]; then
            if ! grep -q "# REDI - Readiness Engine" "$dst" 2>/dev/null; then
                echo "" >> "$dst"
                tail -n +5 "$SKILL_FILE" >> "$dst"
                echo "  Appended SKILL.md -> $dst"
            else
                echo "  REDI skill already in $dst"
            fi
        else
            tail -n +5 "$SKILL_FILE" > "$dst"
            echo "  Created $dst"
        fi
    fi
}

# Install for Codex
install_codex() {
    echo "Installing for Codex..."
    if $GLOBAL; then
        # Global: add to ~/.codex/instructions.md
        local dst="$HOME/.codex/instructions.md"
        mkdir -p "$HOME/.codex"
        if [[ -f "$dst" ]]; then
            if ! grep -q "# REDI - Readiness Engine" "$dst" 2>/dev/null; then
                echo "" >> "$dst"
                tail -n +5 "$SKILL_FILE" >> "$dst"
                echo "  Appended SKILL.md -> $dst"
            else
                echo "  REDI skill already in $dst"
            fi
        else
            tail -n +5 "$SKILL_FILE" > "$dst"
            echo "  Created $dst"
        fi
    else
        # Local: create/update AGENTS.md in project root
        local dst="$PWD/AGENTS.md"
        if [[ -f "$dst" ]]; then
            if ! grep -q "# REDI - Readiness Engine" "$dst" 2>/dev/null; then
                echo "" >> "$dst"
                tail -n +5 "$SKILL_FILE" >> "$dst"
                echo "  Appended SKILL.md -> $dst"
            else
                echo "  REDI skill already in $dst"
            fi
        else
            tail -n +5 "$SKILL_FILE" > "$dst"
            echo "  Created $dst"
        fi
    fi
}

# Main
echo "REDI Skill Installer"
echo "===================="
echo "Source: $SKILL_FILE"
echo "Mode: $($GLOBAL && echo 'global' || echo 'local')"
echo ""

for target in "${TARGETS[@]}"; do
    case "$target" in
        claude) install_claude ;;
        gemini) install_gemini ;;
        codex) install_codex ;;
    esac
done

echo ""
echo "Done!"
