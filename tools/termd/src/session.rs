//! Shared PTY sessions — one long-lived `docker exec` per terminal
//! with input fanned in from every source (WS, HTTP) and output
//! broadcast to every subscriber.
//!
//! The model for a session:
//!
//!   sources ──┐                                 ┌── WS subscribers
//!   (WS, HTTP)│                                 │
//!        ─────┼─► mpsc ──► writer task ─►  PTY  │
//!             │                           stdin │
//!                                               │
//!             ◄── broadcast ◄── reader task ◄── PTY stdout/stderr
//!
//! Sessions are created lazily on first attach so a freshly-spawned
//! container doesn't eat an exec slot until somebody actually opens
//! the terminal. Destroying the container tears the session down.

use anyhow::{anyhow, Context, Result};
use bollard::container::ResizeContainerTtyOptions;
use bollard::exec::{CreateExecOptions, ResizeExecOptions, StartExecOptions, StartExecResults};
use bollard::Docker;
use bytes::Bytes;
use futures_util::StreamExt;
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::io::AsyncWriteExt;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};

use crate::docker::find_container_id;
use crate::state::AppState;

/// How much terminal output to keep for late joiners / `read_terminal`
/// snapshots. ~64 KB is enough for a few screens without blowing up
/// memory even if a hundred sandboxes are hot.
const BACKLOG_BYTES: usize = 64 * 1024;

pub struct Session {
    /// Fan-out of every PTY chunk. Subscribers receive bytes; lagging
    /// subscribers get a `RecvError::Lagged` which the WS side
    /// converts into a quiet re-sync from the backlog.
    pub output: broadcast::Sender<Bytes>,
    /// Input channel — any source can push bytes; a single writer
    /// task drains this into the PTY's stdin.
    pub input: mpsc::Sender<Bytes>,
    /// Backlog ring so a second WS / late subscriber can repaint the
    /// last screenful without replaying from container logs.
    backlog: Arc<Mutex<VecDeque<u8>>>,
    pub exec_id: String,
    pub container_id: String,
}

impl Session {
    pub async fn backlog_snapshot(&self) -> Bytes {
        let buf = self.backlog.lock().await;
        Bytes::from(buf.iter().copied().collect::<Vec<_>>())
    }

    pub async fn send_input(&self, bytes: Bytes) -> Result<()> {
        self.input
            .send(bytes)
            .await
            .map_err(|_| anyhow!("terminal session closed; input dropped"))?;
        Ok(())
    }

    pub async fn resize(
        &self,
        docker: &Docker,
        cols: u16,
        rows: u16,
    ) -> Result<()> {
        let _ = docker
            .resize_exec(
                &self.exec_id,
                ResizeExecOptions {
                    height: rows,
                    width: cols,
                },
            )
            .await;
        let _ = docker
            .resize_container_tty(
                &self.container_id,
                ResizeContainerTtyOptions {
                    height: rows,
                    width: cols,
                },
            )
            .await;
        Ok(())
    }
}

/// Registry of live sessions keyed by the ht-termd terminal id
/// (`TerminalHandle::id`, not the container id — the container id
/// can change across pod restarts if we ever add that feature).
#[derive(Default)]
pub struct SessionRegistry {
    map: RwLock<std::collections::HashMap<String, Arc<Session>>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get-or-create the shared session for a terminal. `create_exec`
    /// is only called when the registry entry is missing; subsequent
    /// attaches return the same `Arc<Session>`.
    pub async fn attach(&self, state: &AppState, terminal_id: &str) -> Result<Arc<Session>> {
        {
            let map = self.map.read().await;
            if let Some(s) = map.get(terminal_id) {
                return Ok(s.clone());
            }
        }

        // Serialise on the write lock so two concurrent attaches
        // don't both spin up their own exec.
        let mut map = self.map.write().await;
        if let Some(s) = map.get(terminal_id) {
            return Ok(s.clone());
        }

        let container_id = find_container_id(state, terminal_id).await?;
        let session = spawn_session(state.docker().clone(), &container_id).await?;
        let arc = Arc::new(session);
        map.insert(terminal_id.to_string(), arc.clone());
        Ok(arc)
    }

    pub async fn get(&self, terminal_id: &str) -> Option<Arc<Session>> {
        self.map.read().await.get(terminal_id).cloned()
    }

    pub async fn remove(&self, terminal_id: &str) {
        self.map.write().await.remove(terminal_id);
    }
}

async fn spawn_session(docker: Docker, container_id: &str) -> Result<Session> {
    // One bash per sandbox — reused across WS attaches. Runs as root
    // so `sudo`, package installs and privileged ops work.
    let exec = docker
        .create_exec(
            container_id,
            CreateExecOptions {
                attach_stdin: Some(true),
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                tty: Some(true),
                cmd: Some(vec!["/bin/bash".to_string()]),
                env: Some(vec!["TERM=xterm-256color".to_string(), "HOME=/root".to_string()]),
                working_dir: Some("/workspace".to_string()),
                ..Default::default()
            },
        )
        .await
        .context("create_exec (session)")?;

    let started = docker
        .start_exec(
            &exec.id,
            Some(StartExecOptions {
                detach: false,
                tty: true,
                output_capacity: None,
            }),
        )
        .await
        .context("start_exec (session)")?;

    let (mut stream, stdin) = match started {
        StartExecResults::Attached { output, input } => (output, input),
        StartExecResults::Detached => {
            return Err(anyhow!("start_exec returned Detached; expected Attached"));
        }
    };

    let (out_tx, _out_rx) = broadcast::channel::<Bytes>(1024);
    let (in_tx, mut in_rx) = mpsc::channel::<Bytes>(256);
    let backlog = Arc::new(Mutex::new(VecDeque::<u8>::with_capacity(BACKLOG_BYTES)));

    // Reader task — push every PTY chunk into the broadcast channel
    // and the backlog ring.
    {
        let out_tx = out_tx.clone();
        let backlog = backlog.clone();
        tokio::spawn(async move {
            while let Some(item) = stream.next().await {
                let Ok(chunk) = item else { break };
                let bytes = Bytes::from(chunk.into_bytes().to_vec());
                if bytes.is_empty() {
                    continue;
                }
                {
                    let mut buf = backlog.lock().await;
                    for b in bytes.iter() {
                        if buf.len() >= BACKLOG_BYTES {
                            buf.pop_front();
                        }
                        buf.push_back(*b);
                    }
                }
                let _ = out_tx.send(bytes);
            }
            tracing::debug!("session reader exited");
        });
    }

    // Writer task — drain every inbound source into PTY stdin. The
    // `stdin` type returned from bollard is a `Pin<Box<dyn AsyncWrite>>`
    // so we just hold it here.
    {
        let mut stdin = stdin;
        tokio::spawn(async move {
            while let Some(bytes) = in_rx.recv().await {
                if stdin.write_all(&bytes).await.is_err() {
                    break;
                }
                let _ = stdin.flush().await;
            }
            let _ = stdin.shutdown().await;
            tracing::debug!("session writer exited");
        });
    }

    Ok(Session {
        output: out_tx,
        input: in_tx,
        backlog,
        exec_id: exec.id,
        container_id: container_id.to_string(),
    })
}
