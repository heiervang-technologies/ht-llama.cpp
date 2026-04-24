# ht-llama-webui on Android

Tauri v2 ships the same webui as the desktop app on Android via
`cargo tauri android build`. The APK auto-targets pre-configured
endpoints so a fresh install drops the user straight into a working
chat — no Settings form to fill in on mobile.

## Pre-configured endpoints (build-time env vars)

The Tauri shell reads three `option_env!("…")` strings at compile
time and injects them as `window.__HT_DEFAULT_*__` globals. The
webui reads the globals as fallbacks whenever the user hasn't set
the corresponding config field themselves.

| Env var                         | Default on Android        | Purpose                                                     |
|---------------------------------|---------------------------|-------------------------------------------------------------|
| `HT_DEFAULT_BACKEND_URL`        | `http://<tailnet-host>:<port>` | llama.cpp endpoint                                          |
| `HT_DEFAULT_TERMINALS_URL`      | `http://<centurion>:43127`    | `ht-termd` sidecar on the desktop host                      |
| `HT_DEFAULT_TERMINALS_TOKEN`    | `<64-hex secret>`             | Bearer for the termd daemon (paired with `--token`)         |

Set them **before** `cargo tauri android build`. They're embedded
into the APK; rebuild when they rotate.

## Network model

The phone reaches the desktop via **tailscale**. The desktop host
runs:

- `llama-server` (or whichever backend the user picked — in practice
  on this user's `rogue` k8s node, reachable on the tailnet as
  `100.88.104.121:30184`).
- `ht-termd` bound to `0.0.0.0:43127` **with a `--token`** so any
  tailnet peer can reach the socket but only callers that know the
  token can spawn shells.

The APK carries the same token, so the phone just works on the
tailnet without the user ever seeing the secret.

## Full build + install

```bash
export ANDROID_HOME=/opt/android-sdk
export NDK_HOME=/opt/android-sdk/ndk/27.2.12479018
export JAVA_HOME=/usr/lib/jvm/default

# Tailnet IPs — run `tailscale status` to find your own.
export HT_DEFAULT_BACKEND_URL="http://100.88.104.121:30184"
export HT_DEFAULT_TERMINALS_URL="http://100.109.19.15:43127"
export HT_DEFAULT_TERMINALS_TOKEN="$(cat ~/.config/ht-termd/token)"

cd tools/server/webui-tauri
cargo tauri android build --debug --apk
adb install -r src-tauri/gen/android/app/build/outputs/apk/universal/debug/app-universal-debug.apk
```

The debug APK is signed with the standard Android debug key, fine
for sideloading. For distribution, follow the Android signing docs
(keystore under `src-tauri/gen/android/keystore.properties`).

## ht-termd systemd user unit

See `tools/termd/ht-termd.service` for a ready-to-drop-in unit.
Put the token in `~/.config/ht-termd/env`:

```
HT_TERMD_BIND=0.0.0.0
HT_TERMD_PORT=43127
HT_TERMD_TOKEN=<the same 64-hex secret the APK was built with>
```

Then:

```bash
cp tools/termd/ht-termd.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ht-termd
```

## Rotating the token

- Change `HT_TERMD_TOKEN` in `~/.config/ht-termd/env`, `systemctl --user restart ht-termd`.
- Rebuild + reinstall the APK with the new `HT_DEFAULT_TERMINALS_TOKEN`.
- Other clients (your desktop Tauri app, web UI) pick up the token
  from `config().terminalsToken` in Settings → Terminals.
