#!/bin/sh
# Install git hooks for the ae4904 workspace.
HOOKS_DIR="$(git rev-parse --git-dir)/hooks"
cat > "$HOOKS_DIR/pre-commit" << 'HOOK'
#!/bin/sh
exec uv run scripts/prek.py
HOOK
chmod +x "$HOOKS_DIR/pre-commit"
echo "pre-commit hook installed."
