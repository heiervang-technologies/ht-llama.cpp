#!/bin/bash

# Script to install pre-commit hook for webui
# Pre-commit: formats, checks, builds webui, stages build output
# Supports HT_TRACKED_DEBT_COMMIT=1 opt-in bypass with audit log

REPO_ROOT=$(git rev-parse --show-toplevel)
PRE_COMMIT_HOOK="$REPO_ROOT/.git/hooks/pre-commit"

echo "Installing pre-commit hook for webui..."

# Create the pre-commit hook
cat > "$PRE_COMMIT_HOOK" << 'EOF'
#!/bin/bash

# Only act when the webui surface is touched.
if ! git diff --cached --name-only | grep -q "^tools/server/webui/"; then
    exit 0
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
WEBUI_DIR="$REPO_ROOT/tools/server/webui"
DEBT_LOG="$REPO_ROOT/.ht-tracked-debt.log"

if [ ! -f "$WEBUI_DIR/package.json" ]; then
    echo "Error: package.json not found in tools/server/webui"
    exit 1
fi

cd "$WEBUI_DIR" || exit 1

# Tracked-debt mode (opt-in): run the same checks, capture the counts,
# but exit 0 regardless so the commit lands. Writes an audit line to
# .ht-tracked-debt.log so a reviewer can see when debt was surfaced
# and how much. Used in place of --no-verify for commits that *intend*
# to land a state where lint/check fail (e.g. when reverting masking
# eslint waivers to expose real Props mismatches).
if [ "$HT_TRACKED_DEBT_COMMIT" = "1" ]; then
    echo "⚠  HT_TRACKED_DEBT_COMMIT=1 — running checks for audit only, not gating" >&2

    lint_out=$(npm run lint 2>&1)
    lint_errs=$(echo "$lint_out" | grep -cE '^\s*[0-9]+:[0-9]+\s+error' || true)

    check_out=$(npm run check 2>&1)
    check_errs=$(echo "$check_out" | grep -oE 'found [0-9]+ errors?' | head -1 | grep -oE '[0-9]+' || echo 0)

    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    branch=$(git rev-parse --abbrev-ref HEAD)
    files=$(git diff --cached --name-only | wc -l | tr -d ' ')
    printf '%s\tbranch=%s\tstaged=%s\tlint=%s\tcheck=%s\n' \
        "$ts" "$branch" "$files" "$lint_errs" "$check_errs" >> "$DEBT_LOG"

    echo "⚠  tracked-debt audit logged: lint=$lint_errs check=$check_errs → $DEBT_LOG" >&2
    echo "⚠  unset HT_TRACKED_DEBT_COMMIT for the next commit to re-engage gating" >&2
    exit 0
fi

# Normal gating mode.

echo "Formatting and checking webui code..."

npm run format || { echo "Error: npm run format failed"; exit 1; }
npm run lint   || { echo "Error: npm run lint failed";   exit 1; }
npm run check  || { echo "Error: npm run check failed";  exit 1; }

echo "✅ Webui code formatted and checked successfully"

echo "Building webui..."
npm run build || { echo "❌ npm run build failed"; exit 1; }

# Stage the build output alongside the source changes so the bundled
# assets travel with the same commit that produced them.
cd "$REPO_ROOT" || exit 1
git add tools/server/public/ 2>/dev/null || true

echo "✅ Webui built and build output staged"
exit 0
EOF

# Make hook executable
chmod +x "$PRE_COMMIT_HOOK"

if [ $? -eq 0 ]; then
    echo "✅ Git hook installed successfully!"
    echo "   Pre-commit: $PRE_COMMIT_HOOK"
    echo ""
    echo "The hook will automatically:"
    echo "  • Format, lint, and check webui code before commits"
    echo "  • Build webui and stage the bundled assets"
    echo ""
    echo "To land a commit that intentionally surfaces lint/check debt"
    echo "(e.g. removing masking eslint waivers), set the env var:"
    echo "  HT_TRACKED_DEBT_COMMIT=1 git commit -m '...'"
    echo "The hook will then log error counts to .ht-tracked-debt.log"
    echo "and exit 0, leaving the gate visible via the audit trail."
else
    echo "❌ Failed to make hook executable"
    exit 1
fi
