//! WebSocket bridge from the browser's xterm.js to the shared PTY
//! session (see `session.rs`). Multiple WS connections on the same
//! terminal id share a single bash — fan-in inputs, fan-out outputs.
//!
//! Protocol: **binary** frames are raw PTY bytes in both directions.
//! **Text** frames are JSON control messages:
//!   `{"t":"resize","cols":N,"rows":N}`
//! Unknown text is forwarded to stdin verbatim so legacy clients
//! that send keystrokes as text still work.

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket};
use bytes::Bytes;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::broadcast::error::RecvError;

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
    socket: WebSocket,
) -> Result<()> {
    let session = state.sessions().attach(&state, terminal_id).await?;
    let (mut sink, mut stream) = socket.split();

    // Replay the backlog to this subscriber so a late joiner or a
    // mode-switch remount sees the last screen worth of output.
    let backlog = session.backlog_snapshot().await;
    if !backlog.is_empty() {
        sink.send(Message::Binary(backlog.to_vec())).await.ok();
    }

    let mut out_rx = session.output.subscribe();
    let mut running = true;

    while running {
        tokio::select! {
            // Shared PTY → this WS
            msg = out_rx.recv() => {
                match msg {
                    Ok(bytes) => {
                        if sink.send(Message::Binary(bytes.to_vec())).await.is_err() { break; }
                    }
                    Err(RecvError::Lagged(_)) => {
                        // Fell behind — repaint the backlog so the
                        // subscriber isn't left with a gap, then
                        // keep reading.
                        let snap = session.backlog_snapshot().await;
                        if sink.send(Message::Binary(snap.to_vec())).await.is_err() { break; }
                    }
                    Err(RecvError::Closed) => break,
                }
            }
            // This WS → shared PTY
            incoming = stream.next() => {
                match incoming {
                    Some(Ok(Message::Binary(bytes))) => {
                        session.send_input(Bytes::from(bytes)).await.ok();
                    }
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(frame) = serde_json::from_str::<ControlFrame>(&text) {
                            match frame {
                                ControlFrame::Resize { cols, rows } => {
                                    let _ = session.resize(state.docker(), cols, rows).await;
                                }
                            }
                        } else {
                            session.send_input(Bytes::from(text.into_bytes())).await.ok();
                        }
                    }
                    Some(Ok(Message::Ping(_)))
                    | Some(Ok(Message::Pong(_))) => { /* axum handles keepalive */ }
                    Some(Ok(Message::Close(_))) => { running = false; }
                    Some(Err(_)) | None => { running = false; }
                }
            }
        }
    }
    Ok(())
}
