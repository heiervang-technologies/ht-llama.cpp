# ht-termd

Terminal-sandbox sidecar for the ht-llama.cpp webui. Spawns
gVisor-hardened containers (via the existing `unleash sandbox`
tooling's `unleash-sandbox` Docker network) and bridges xterm.js
WebSockets to `docker exec -it bash` inside them.

## Security invariants

Every container-create path refuses unless **all** four of the
following are true at call time:

1. `docker info` advertises `runsc` among its runtimes.
2. The sandbox network exists **and** has
   `com.docker.network.bridge.enable_icc=false`.
3. iptables `DOCKER-USER` contains DROP rules for 10/8, 172.16/12,
   192.168/16, 169.254/16.
4. The target image exists locally.

Run `unleash sandbox setup` on the host once; ht-termd will then
happily start.

## HTTP surface

| Route                       | Verb      | Purpose                                      |
|-----------------------------|-----------|----------------------------------------------|
| `/health`                   | GET       | liveness + sandbox readiness flags           |
| `/v1/sandbox/status`        | GET       | structured readiness breakdown               |
| `/v1/terminals`             | GET       | list sandboxes we own                        |
| `/v1/terminals`             | POST      | create a new sandbox (body: `{name?}`)       |
| `/v1/terminals/:id`         | DELETE    | destroy a sandbox + wipe its scratch volume  |
| `/v1/terminals/:id/ws`      | GET (WS)  | attach an xterm.js-style shell               |

WebSocket protocol: **binary** frames are raw container bytes in
both directions. **Text** frames are JSON control messages — today
only `{"t":"resize","cols":N,"rows":N}`. Unknown text is forwarded
to container stdin.

## Build & run

```bash
cargo build --release -p ht-termd
./target/release/ht-termd --bind 127.0.0.1 --port 43127
```

All flags also read from `HT_TERMD_*` env vars. The binary refuses
to start with any runtime other than `runsc`.

## Filesystem layout

Each sandbox gets its own scratch volume mounted as `/workspace`,
kept under `$XDG_STATE_HOME/ht-termd/workspaces/<id>` on the host
(falling back to `$HOME/.local/state/...`). The volume persists
across tab close / WS detach; `DELETE /v1/terminals/:id` wipes it.

## Ownership

Every container we create carries the label
`heiervang.ht-termd=true`. List/delete paths filter on that label
so we never rm somebody else's container.
