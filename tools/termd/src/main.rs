//! ht-termd — terminal-sandbox sidecar.
//!
//! HTTP + WebSocket front-end that:
//! 1. Spawns gVisor-hardened `unleash:latest` containers on the
//!    `unleash-sandbox` Docker network (internet yes, LAN no).
//! 2. Bridges xterm.js WebSockets to `docker exec -it bash` streams
//!    inside those containers.
//! 3. Enforces hard refusal gates: no `runsc` runtime, no
//!    `unleash-sandbox` network, or missing iptables LAN-drop rules
//!    ⇒ refuses to spawn.
//!
//! Binds loopback by default. Meant to run locally alongside
//! `llama-server` (either as a Tauri-managed child process or as a
//! standalone systemd unit).

mod docker;
mod http;
mod sandbox_guard;
mod session;
mod state;
mod ws;

use anyhow::Result;
use clap::Parser;
use std::net::{IpAddr, SocketAddr};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Debug, Parser)]
#[command(
    name = "ht-termd",
    version,
    about = "Terminal-sandbox sidecar for the ht-llama.cpp webui."
)]
struct Args {
    /// Bind address. Defaults to loopback; override carefully — the
    /// API has no auth and hands out shells.
    #[arg(long, default_value = "127.0.0.1", env = "HT_TERMD_BIND")]
    bind: IpAddr,

    /// Port to listen on.
    #[arg(long, default_value_t = 43127, env = "HT_TERMD_PORT")]
    port: u16,

    /// Docker image used for new sandboxes.
    #[arg(long, default_value = "unleash:latest", env = "HT_TERMD_IMAGE")]
    image: String,

    /// Docker network the sandboxes attach to. Must exist and must
    /// already have the LAN-drop iptables rules applied (run
    /// `unleash sandbox setup` once before starting this daemon).
    #[arg(long, default_value = "unleash-sandbox", env = "HT_TERMD_NETWORK")]
    network: String,

    /// Container runtime. Must be `runsc` — refuses to start
    /// otherwise.
    #[arg(long, default_value = "runsc", env = "HT_TERMD_RUNTIME")]
    runtime: String,

    /// Shared-secret bearer token. When set, clients must pass it as
    /// `Authorization: Bearer <token>` on HTTP, or `?token=<token>`
    /// on the WS upgrade. Leave unset for loopback-only deployments
    /// where the network already authenticates. Binding non-loopback
    /// without a token logs a loud warning but does not refuse to
    /// start — your firewall may already restrict access.
    #[arg(long, env = "HT_TERMD_TOKEN")]
    token: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer().with_target(false))
        .init();

    let args = Args::parse();
    if args.runtime != "runsc" {
        anyhow::bail!(
            "runtime must be 'runsc' (gVisor); refusing to start with '{}'",
            args.runtime
        );
    }

    let addr = SocketAddr::new(args.bind, args.port);

    // Binding non-loopback without a token is a foot-gun: any peer
    // that can reach the socket gets a root shell in a sandbox. Log
    // a loud warning so the operator notices even if they blew past
    // the `--help` text.
    if !args.bind.is_loopback() && args.token.is_none() {
        tracing::warn!(
            bind = %args.bind,
            "ht-termd is binding to a non-loopback interface WITHOUT a --token; \
             any reachable peer can spawn a shell. Set --token (or HT_TERMD_TOKEN) \
             for Tailscale / LAN deployments."
        );
    }

    let token_set = args.token.is_some();
    let state = state::AppState::new(args.image.clone(), args.network.clone(), args.token).await?;

    let app = http::router(state.clone());
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, image = %args.image, network = %args.network, auth = token_set, "ht-termd listening");
    axum::serve(listener, app).await?;
    Ok(())
}
