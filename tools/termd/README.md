# ht-termd

Terminal-sandbox sidecar for heierchat tool execution. Spawns
gVisor-hardened containers (via the existing `unleash sandbox`
tooling's `unleash-sandbox` Docker network) and bridges xterm.js
WebSockets to `docker exec -it bash` inside them.

## Security invariants

Every container-create path refuses unless **all** four of the
following are true at call time:

1. `docker info` advertises `runsc` among its runtimes.
2. The sandbox network exists **and** has
   `com.docker.network.bridge.enable_icc=false`.
3. iptables `DOCKER-USER` contains DROP rules, scoped to the sandbox
   network subnet, for 10/8, 172.16/12, 192.168/16, 169.254/16.
4. The target image exists locally.

Run `unleash sandbox setup` on the host once; ht-termd will then
happily start. The service account must be able to verify the chain
with either `iptables -S DOCKER-USER` or passwordless
`sudo -n iptables -S DOCKER-USER`; an unverifiable chain fails closed.

## HTTP surface

| Route                              | Verb      | Purpose                                      |
|------------------------------------|-----------|----------------------------------------------|
| `/health`                          | GET       | liveness + sandbox readiness flags           |
| `/v1/sandbox/status`               | GET       | structured readiness breakdown               |
| `/v1/terminals`                    | GET       | list sandboxes we own                        |
| `/v1/terminals`                    | POST      | create a new sandbox (body below)            |
| `/v1/terminals/:id`                | DELETE    | destroy a sandbox + wipe its scratch volume  |
| `/v1/terminals/:id/input`          | POST      | inject keystrokes (text / base64 / auto-Enter) into the shared PTY |
| `/v1/terminals/:id/bootstrap-log`  | GET       | read the per-terminal bootstrap stdout/stderr |
| `/v1/terminals/:id/ws`             | GET (WS)  | attach an xterm.js-style shell               |

`POST /v1/terminals` body fields (all optional):

```jsonc
{
  "name": "string",        // display name
  "bootstrap": "string",   // shell snippet that runs once as root after files
                           // have been written; output captured in the
                           // bootstrap log
  "env": { "K": "V" },     // extra env vars for every docker-exec invocation
  "files": [               // files dropped before the bootstrap runs
    { "path": "/workspace/foo", "content": "...", "mode": 420 }
    // prefix `content` with `base64:` for binary payloads
  ]
}
```

WebSocket protocol: **binary** frames are raw container bytes in
both directions. **Text** frames are JSON control messages —
`{"t":"resize","cols":N,"rows":N}` and `{"t":"ping"}` (no-op,
keepalive). Unknown text is forwarded to container stdin.

## Authentication

If started with `--token <T>` (or `HT_TERMD_TOKEN=…`), every HTTP
request must carry `Authorization: Bearer <T>` and every WebSocket
upgrade must include `?token=<T>` in the query (browsers don't
allow custom headers on `new WebSocket()`). Without `--token` the
daemon accepts loopback callers only. A non-loopback bind without a
token is refused at startup.

## Build & run

```bash
cargo build --release -p ht-termd
./target/release/ht-termd --bind 127.0.0.1 --port 43127 --token "$(openssl rand -hex 32)"
```

All flags also read from `HT_TERMD_*` env vars. The binary refuses
to start with any runtime other than `runsc`.

A reference systemd user unit lives at `tools/termd/ht-termd.service`.

## Filesystem layout

Each sandbox gets its own scratch volume mounted as `/workspace`,
kept under `$XDG_STATE_HOME/ht-termd/workspaces/<id>` on the host
(falling back to `$HOME/.local/state/...`). The volume persists
across tab close / WS detach; `DELETE /v1/terminals/:id` wipes it.

## Ownership

Every container we create carries the label
`heiervang.ht-termd=true`. List/delete paths filter on that label
so we never rm somebody else's container.
