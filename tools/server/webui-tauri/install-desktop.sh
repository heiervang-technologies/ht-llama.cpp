#!/usr/bin/env bash
# Install ht-llama-webui desktop entries + icon into the user's XDG dirs.
# Rewrites Exec paths so the dev launcher works from any directory.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICONS="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"

mkdir -p "$APPS"
for size in 32 128 256; do
    mkdir -p "$ICONS/${size}x${size}/apps"
done

# Pick the best available source icon for each size.
install -m 0644 "$DIR/src-tauri/icons/32x32.png"       "$ICONS/32x32/apps/ht-llama-webui.png"
install -m 0644 "$DIR/src-tauri/icons/128x128.png"     "$ICONS/128x128/apps/ht-llama-webui.png"
install -m 0644 "$DIR/src-tauri/icons/128x128@2x.png"  "$ICONS/256x256/apps/ht-llama-webui.png"

# Rewrite the dev launcher so %k (the .desktop file's own dir) resolves to the
# installed copy — which isn't in the repo. We bake the repo path in directly.
sed "s|%k/\\.\\.|$DIR|g" "$DIR/ht-llama-webui.desktop"     > "$APPS/ht-llama-webui.desktop"
sed "s|%k/\\.\\.|$DIR|g" "$DIR/ht-llama-webui-dev.desktop" > "$APPS/ht-llama-webui-dev.desktop"
chmod 0644 "$APPS/ht-llama-webui.desktop" "$APPS/ht-llama-webui-dev.desktop"

command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" || true
command -v gtk-update-icon-cache    >/dev/null && gtk-update-icon-cache -q "$ICONS" 2>/dev/null || true

echo "Installed:"
echo "  $APPS/ht-llama-webui.desktop"
echo "  $APPS/ht-llama-webui-dev.desktop"
echo "  $ICONS/{32,128,256}x.../apps/ht-llama-webui.png"
