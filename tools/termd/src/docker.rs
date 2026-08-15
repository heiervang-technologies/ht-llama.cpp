//! Docker container lifecycle for sandboxes.
//!
//! Each sandbox is a long-running container started from
//! `unleash:latest` under the `runsc` runtime on the
//! `unleash-sandbox` network. The entrypoint is `tail -f /dev/null`
//! so the container stays alive across WS attach/detach cycles; the
//! actual shell is spun up via `docker exec` in `ws.rs` on demand.

use anyhow::{anyhow, Context, Result};
use bollard::container::{
    Config, CreateContainerOptions, ListContainersOptions, RemoveContainerOptions,
    StartContainerOptions,
};
use bollard::models::{HostConfig, Mount, MountTypeEnum, RestartPolicy, RestartPolicyNameEnum};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

use crate::state::AppState;

/// The label we stamp on every sandbox container we own. Used for
/// `list` filtering and to refuse to rm/resize containers that
/// weren't created by us.
pub const OWNER_LABEL: &str = "heiervang.ht-termd";
pub const OWNER_VALUE: &str = "true";

#[derive(Debug, Clone, Serialize)]
pub struct TerminalHandle {
    pub id: String,
    pub name: String,
    pub container_id: String,
    pub image: String,
    pub status: String,
    pub created_at: i64,
}

#[derive(Debug, Deserialize, Default)]
pub struct CreateBody {
    /// Human-friendly name. Becomes the container hostname and the
    /// `name` label. Uniqueness not enforced — two sandboxes can
    /// share a name.
    pub name: Option<String>,
    /// Extra env vars injected on top of the default TERM/HOME.
    /// Applied at container-create time so every `docker exec` sees
    /// them (including the session bash in `session.rs`).
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    /// Files to drop into the container before bootstrap runs.
    /// `content` may be a plain UTF-8 string or a `base64:…` payload
    /// for binary. Paths are interpreted inside the container —
    /// absolute or relative to `/workspace`. Mode defaults to 0644.
    #[serde(default)]
    pub files: Vec<CreateFile>,
    /// Bash script that runs **once**, as root, inside the container
    /// after files have been written. Exit code is ignored; stdout
    /// / stderr are captured and surfaced via `GET
    /// /v1/terminals/:id/bootstrap-log`.
    pub bootstrap: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct CreateFile {
    pub path: String,
    pub content: String,
    #[serde(default)]
    pub mode: Option<u32>,
}

pub async fn list_terminals(state: &AppState) -> Result<Vec<TerminalHandle>> {
    let mut filters = HashMap::new();
    filters.insert(
        "label".to_string(),
        vec![format!("{OWNER_LABEL}={OWNER_VALUE}")],
    );
    let opts = ListContainersOptions {
        all: true,
        filters,
        ..Default::default()
    };
    let containers = state.docker().list_containers(Some(opts)).await?;
    Ok(containers
        .into_iter()
        .filter_map(|c| {
            let labels = c.labels.unwrap_or_default();
            let id = labels.get("heiervang.ht-termd.id")?.clone();
            let name = labels
                .get("heiervang.ht-termd.name")
                .cloned()
                .unwrap_or_else(|| id.clone());
            Some(TerminalHandle {
                id,
                name,
                container_id: c.id.unwrap_or_default(),
                image: c.image.unwrap_or_default(),
                status: c.state.unwrap_or_default(),
                created_at: c.created.unwrap_or_default(),
            })
        })
        .collect())
}

pub async fn create_terminal(state: &AppState, body: CreateBody) -> Result<TerminalHandle> {
    let id = Uuid::new_v4().to_string();
    let mut name = body
        .name
        .unwrap_or_else(|| format!("term-{}", &id[..8]))
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(48)
        .collect::<String>();
    if name.is_empty() {
        name = format!("term-{}", &id[..8]);
    }
    let container_name = format!("ht-termd-{}", &id[..8]);

    // Per-terminal scratch volume. Persists across attach/detach so
    // dotfiles / half-done work survives a tab close, but gets
    // wiped when the user explicitly deletes the terminal.
    let host_scratch = scratch_path(&id)?;
    std::fs::create_dir_all(&host_scratch)
        .with_context(|| format!("create scratch dir {host_scratch:?}"))?;

    let mut labels: HashMap<String, String> = HashMap::new();
    labels.insert(OWNER_LABEL.to_string(), OWNER_VALUE.to_string());
    labels.insert("heiervang.ht-termd.id".to_string(), id.clone());
    labels.insert("heiervang.ht-termd.name".to_string(), name.clone());

    let host_config = HostConfig {
        runtime: Some("runsc".to_string()),
        network_mode: Some(state.network().to_string()),
        dns: Some(vec!["8.8.8.8".to_string(), "8.8.4.4".to_string()]),
        restart_policy: Some(RestartPolicy {
            name: Some(RestartPolicyNameEnum::NO),
            maximum_retry_count: None,
        }),
        mounts: Some(vec![Mount {
            target: Some("/workspace".to_string()),
            source: Some(host_scratch.to_string_lossy().into_owned()),
            typ: Some(MountTypeEnum::BIND),
            read_only: Some(false),
            ..Default::default()
        }]),
        auto_remove: Some(false),
        // Defensive caps — keep the set small, but include what
        // sudo + package managers inside the image actually need.
        // The real sandbox boundary is gVisor (runsc) plus the
        // LAN-block iptables rules on the unleash-sandbox network;
        // stripping caps below this point doesn't add security, it
        // just breaks basic shell operations.
        //
        // Deliberately NOT setting `no-new-privileges:true` — that
        // would block sudo's setuid transition even though sudo is
        // installed, which was surprising and wrong for an
        // interactive sandbox.
        cap_drop: Some(vec!["ALL".to_string()]),
        cap_add: Some(vec![
            "CHOWN".to_string(),
            "DAC_OVERRIDE".to_string(),
            "FOWNER".to_string(),
            "FSETID".to_string(),
            "SETUID".to_string(),
            "SETGID".to_string(),
            "SETPCAP".to_string(),
            "NET_BIND_SERVICE".to_string(),
            "KILL".to_string(),
            "AUDIT_WRITE".to_string(),
        ]),
        ..Default::default()
    };

    let mut env = vec!["HOME=/root".to_string()];
    for (k, v) in &body.env {
        // Silently skip malformed keys rather than 400-ing the
        // whole create — partial env is still useful.
        if k.is_empty() || k.contains('=') {
            continue;
        }
        env.push(format!("{k}={v}"));
    }

    let config = Config {
        image: Some(state.image().to_string()),
        user: Some("0:0".to_string()),
        hostname: Some(name.clone()),
        tty: Some(true),
        open_stdin: Some(true),
        attach_stdin: Some(false),
        attach_stdout: Some(false),
        attach_stderr: Some(false),
        working_dir: Some("/workspace".to_string()),
        env: Some(env),
        // Keep the container alive via an idle command. We set
        // `cmd` rather than `entrypoint` so the image's default
        // `entrypoint.sh` runs first — under gVisor Docker's
        // internal DNS resolver (127.0.0.11) doesn't work, and that
        // script rewrites `/etc/resolv.conf` to public nameservers.
        // Overriding `entrypoint` here would skip the DNS fix and
        // container hostname lookups would silently fail.
        cmd: Some(vec![
            "tail".to_string(),
            "-f".to_string(),
            "/dev/null".to_string(),
        ]),
        labels: Some(labels),
        host_config: Some(host_config),
        ..Default::default()
    };

    let create_opts = CreateContainerOptions {
        name: container_name.clone(),
        platform: None,
    };
    let created = match state
        .docker()
        .create_container(Some(create_opts), config)
        .await
    {
        Ok(created) => created,
        Err(err) => {
            let _ = std::fs::remove_dir_all(&host_scratch);
            return Err(err).context("docker create");
        }
    };
    if let Err(err) = state
        .docker()
        .start_container(&created.id, None::<StartContainerOptions<String>>)
        .await
    {
        let _ = state
            .docker()
            .remove_container(
                &created.id,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await;
        let _ = std::fs::remove_dir_all(&host_scratch);
        return Err(err).context("docker start");
    }

    // Write any user-provided files, then run the bootstrap script.
    // Everything runs as root inside the container via docker exec;
    // failures are logged but don't fail the create — the sandbox
    // still comes up, the bootstrap log captures the error, the
    // user can re-run the snippet interactively.
    if !body.files.is_empty() {
        if let Err(err) = write_files(state, &created.id, &body.files).await {
            tracing::warn!(id = %id, error = %err, "bootstrap: file write failed");
        }
    }
    if let Some(script) = body.bootstrap.as_deref() {
        if !script.trim().is_empty() {
            if let Err(err) = run_bootstrap(state, &created.id, &id, script).await {
                tracing::warn!(id = %id, error = %err, "bootstrap: script failed");
            }
        }
    }

    Ok(TerminalHandle {
        id,
        name,
        container_id: created.id,
        image: state.image().to_string(),
        status: "running".to_string(),
        created_at: chrono_ts_millis(),
    })
}

/// Write each file payload to its `path` inside the container. Text
/// goes in verbatim; a `base64:` prefix marks a binary blob that
/// gets decoded first. Parent dirs are `mkdir -p`'d automatically.
async fn write_files(state: &AppState, container_id: &str, files: &[CreateFile]) -> Result<()> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
    use futures_util::StreamExt;

    for f in files {
        if f.path.is_empty() || f.path.contains('\0') {
            return Err(anyhow!(
                "file path must be non-empty and contain no NUL bytes"
            ));
        }
        let path = if f.path.starts_with('/') {
            f.path.clone()
        } else {
            format!("/workspace/{}", f.path)
        };
        let bytes = if let Some(b64) = f.content.strip_prefix("base64:") {
            STANDARD.decode(b64.trim()).context("base64 decode")?
        } else {
            f.content.as_bytes().to_vec()
        };
        let b64 = STANDARD.encode(&bytes);
        let mode = f.mode.unwrap_or(0o644);
        if mode > 0o7777 {
            return Err(anyhow!("file mode {mode:#o} exceeds 0o7777"));
        }
        // Pass all user-controlled values as argv entries. This keeps
        // paths such as `a'b` from changing the shell program.
        let script = "set -e; path=$1; payload=$2; mode=$3; mkdir -p \"$(dirname -- \"$path\")\"; printf '%s' \"$payload\" | base64 -d > \"$path\"; chmod \"$mode\" \"$path\"";
        let exec = state
            .docker()
            .create_exec(
                container_id,
                CreateExecOptions {
                    attach_stdout: Some(true),
                    attach_stderr: Some(true),
                    cmd: Some(vec![
                        "/bin/bash".to_string(),
                        "-c".to_string(),
                        script.to_string(),
                        "ht-termd-write".to_string(),
                        path,
                        b64,
                        format!("{mode:o}"),
                    ]),
                    ..Default::default()
                },
            )
            .await
            .context("create_exec (files)")?;
        let started = state
            .docker()
            .start_exec(
                &exec.id,
                Some(StartExecOptions {
                    detach: false,
                    tty: false,
                    output_capacity: None,
                }),
            )
            .await
            .context("start_exec (files)")?;
        let StartExecResults::Attached { mut output, .. } = started else {
            return Err(anyhow!("file write exec detached unexpectedly"));
        };
        while let Some(chunk) = output.next().await {
            chunk.context("file write exec output")?;
        }
        let inspected = state
            .docker()
            .inspect_exec(&exec.id)
            .await
            .context("inspect_exec (files)")?;
        if inspected.exit_code != Some(0) {
            return Err(anyhow!(
                "file write exec exited with status {:?}",
                inspected.exit_code
            ));
        }
    }
    Ok(())
}

/// Run a bootstrap script with stdout/stderr captured to a known
/// path inside the container (`/var/log/ht-termd-bootstrap.log`).
/// Uses base64 passthrough so users can paste any script without
/// worrying about quote escaping.
async fn run_bootstrap(
    state: &AppState,
    container_id: &str,
    terminal_id: &str,
    script: &str,
) -> Result<()> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    use bollard::exec::{CreateExecOptions, StartExecOptions};

    let encoded = STANDARD.encode(script.as_bytes());
    let wrapper = format!(
        "mkdir -p /var/log; echo '--- ht-termd bootstrap for {terminal_id} ---' > /var/log/ht-termd-bootstrap.log; echo {encoded} | base64 -d > /tmp/ht-termd-bootstrap.sh; chmod +x /tmp/ht-termd-bootstrap.sh; bash /tmp/ht-termd-bootstrap.sh >> /var/log/ht-termd-bootstrap.log 2>&1 & echo $! > /tmp/ht-termd-bootstrap.pid"
    );
    let exec = state
        .docker()
        .create_exec(
            container_id,
            CreateExecOptions {
                attach_stdout: Some(false),
                attach_stderr: Some(false),
                cmd: Some(vec!["/bin/bash".to_string(), "-c".to_string(), wrapper]),
                ..Default::default()
            },
        )
        .await
        .context("create_exec (bootstrap)")?;
    state
        .docker()
        .start_exec(
            &exec.id,
            Some(StartExecOptions {
                detach: true,
                tty: false,
                output_capacity: None,
            }),
        )
        .await
        .context("start_exec (bootstrap)")?;
    Ok(())
}

/// Read the bootstrap log out of the container, if present. Returns
/// empty string if the terminal had no bootstrap or the file hasn't
/// been created yet.
pub async fn read_bootstrap_log(state: &AppState, terminal_id: &str) -> Result<String> {
    use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};

    let container_id = find_container_id(state, terminal_id).await?;
    let exec = state
        .docker()
        .create_exec(
            &container_id,
            CreateExecOptions {
                attach_stdout: Some(true),
                attach_stderr: Some(true),
                cmd: Some(vec![
                    "/bin/bash".to_string(),
                    "-c".to_string(),
                    "cat /var/log/ht-termd-bootstrap.log 2>/dev/null || true".to_string(),
                ]),
                ..Default::default()
            },
        )
        .await?;
    let started = state
        .docker()
        .start_exec(
            &exec.id,
            Some(StartExecOptions {
                detach: false,
                tty: false,
                output_capacity: None,
            }),
        )
        .await?;
    let StartExecResults::Attached { mut output, .. } = started else {
        return Ok(String::new());
    };
    use futures_util::StreamExt;
    let mut buf = Vec::new();
    while let Some(chunk) = output.next().await {
        if let Ok(c) = chunk {
            buf.extend_from_slice(&c.into_bytes());
        }
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

pub async fn delete_terminal(state: &AppState, id: &str) -> Result<()> {
    let container_id = find_container_id(state, id).await?;
    let opts = RemoveContainerOptions {
        force: true,
        v: true,
        ..Default::default()
    };
    state
        .docker()
        .remove_container(&container_id, Some(opts))
        .await
        .context("docker rm")?;

    // Drop the shared session so the next terminal with the same id
    // (extremely unlikely — ids are uuids — but correct to do) gets
    // a fresh exec.
    state.sessions().remove(id).await;

    // Wipe the scratch volume. Best-effort — if the directory was
    // already cleaned up externally, that's fine.
    let _ = std::fs::remove_dir_all(scratch_path(id)?);
    Ok(())
}

pub async fn find_container_id(state: &AppState, id: &str) -> Result<String> {
    Uuid::parse_str(id).with_context(|| format!("invalid terminal id {id:?}"))?;
    let mut filters = HashMap::new();
    filters.insert(
        "label".to_string(),
        vec![
            format!("{OWNER_LABEL}={OWNER_VALUE}"),
            format!("heiervang.ht-termd.id={id}"),
        ],
    );
    let opts = ListContainersOptions {
        all: true,
        filters,
        ..Default::default()
    };
    let containers = state.docker().list_containers(Some(opts)).await?;
    containers
        .into_iter()
        .next()
        .and_then(|c| c.id)
        .ok_or_else(|| anyhow!("terminal {id} not found"))
}

fn scratch_path(id: &str) -> Result<PathBuf> {
    let base = std::env::var("XDG_STATE_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".local").join("state"))
        })
        .ok_or_else(|| anyhow!("cannot determine XDG_STATE_HOME / HOME for scratch dir"))?;
    Ok(base.join("ht-termd").join("workspaces").join(id))
}

fn chrono_ts_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
