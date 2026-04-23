//! WebSocket ↔ `docker exec -it bash` bridge.
//!
//! Protocol: raw container bytes in both directions for stdin/stdout.
//! Control frames are JSON text messages the webui sends for
//! out-of-band signals — currently just `{"t":"resize","cols":N,"rows":N}`.
//! Everything else is binary and passed through verbatim.

use anyhow::{anyhow, Context, Result};
use axum::extract::ws::{Message, WebSocket};
use bollard::container::ResizeContainerTtyOptions;
use bollard::exec::{CreateExecOptions, ResizeExecOptions, StartExecOptions, StartExecResults};
use bollard::Docker;
use futures_util::StreamExt;
use serde::Deserialize;
use tokio::io::AsyncWriteExt;

use crate::docker::find_container_id;
use crate::state::AppState;

#[derive(Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
enum ControlFrame {
    Resize { cols: u16, rows: u16 },
}

pub async fn bridge(state: AppState, terminal_id: String, socket: WebSocket) {
    if let Err(err) = run_bridge(state, &terminal_id, socket).await {
        tracing::warn!(id = %terminal_id, error = %err, "ws bridge ended with error");
    }
}

async fn run_bridge(
    state: AppState,
    terminal_id: &str,
    mut socket: WebSocket,
) -> Result<()> {
    let container_id = find_container_id(&state, terminal_id).await?;
    let docker = state.docker().clone();

    let exec = docker
        .create_exec(
            &container_id,
            CreateExecOptions {
                attach_stdin: Some(true),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                tty: Some(true),
                cmd: Some(vec!["/bin/bash".to_string()]),
                env: Some(vec![
                    "TERM=xterm-256color".to_string(),
                    "HOME=/home/unleash".to_string(),
                ]),
                working_dir: Some("/workspace".to_string()),
                user: Some("unleash".to_string()),
                ..Default::default()
            },
        )
        .await
        .context("create_exec")?;

    let started = docker
        .start_exec(&exec.id, Some(StartExecOptions { detach: false, tty: true, output_capacity: None }))
        .await
        .context("start_exec")?;

    let (mut output, mut stdin) = match started {
        StartExecResults::Attached { output, input } => (output, input),
        StartExecResults::Detached => {
            return Err(anyhow!("start_exec returned Detached; expected Attached"))
        }
    };

    loop {
        tokio::select! {
            // Container → WebSocket
            next = output.next() => {
                match next {
                    Some(Ok(chunk)) => {
                        let bytes = chunk.into_bytes();
                        if bytes.is_empty() { continue; }
                        if socket.send(Message::Binary(bytes.to_vec())).await.is_err() {
                            break;
                        }
                    }
                    Some(Err(err)) => {
                        tracing::debug!(%err, "exec output stream error");
                        break;
                    }
                    None => break,
                }
            }
            // WebSocket → container / control
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Binary(bytes))) => {
                        if stdin.write_all(&bytes).await.is_err() { break; }
                    }
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(frame) = serde_json::from_str::<ControlFrame>(&text) {
                            handle_control(&docker, &container_id, &exec.id, frame).await;
                        } else {
                            // Legacy clients may send raw text; forward as stdin.
                            if stdin.write_all(text.as_bytes()).await.is_err() { break; }
                        }
                    }
                    Some(Ok(Message::Ping(_)))
                    | Some(Ok(Message::Pong(_))) => { /* axum handles keepalive */ }
                    Some(Ok(Message::Close(_))) => break,
                    Some(Err(_)) | None => break,
                }
            }
        }
    }

    // Best-effort: let the remote shell flush on its own; we don't
    // kill the exec because the container is still owned by the
    // sandbox and a fresh attach can get a new bash on demand.
    let _ = stdin.shutdown().await;
    Ok(())
}

async fn handle_control(
    docker: &Docker,
    container_id: &str,
    exec_id: &str,
    frame: ControlFrame,
) {
    match frame {
        ControlFrame::Resize { cols, rows } => {
            // Resize both the exec PTY (so bash sees SIGWINCH) and
            // the container's TTY (keeps tools that peek at the
            // container-level size happy).
            let _ = docker
                .resize_exec(
                    exec_id,
                    ResizeExecOptions {
                        height: rows,
                        width: cols,
                    },
                )
                .await;
            let _ = docker
                .resize_container_tty(
                    container_id,
                    ResizeContainerTtyOptions {
                        height: rows,
                        width: cols,
                    },
                )
                .await;
        }
    }
}
