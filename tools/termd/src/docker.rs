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

pub async fn create_terminal(
    state: &AppState,
    body: CreateBody,
) -> Result<TerminalHandle> {
    let id = Uuid::new_v4().to_string();
    let name = body
        .name
        .unwrap_or_else(|| format!("term-{}", &id[..8]))
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .take(48)
        .collect::<String>();
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

    let env = vec!["HOME=/home/unleash".to_string()];

    let config = Config {
        image: Some(state.image().to_string()),
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
        cmd: Some(vec!["tail".to_string(), "-f".to_string(), "/dev/null".to_string()]),
        labels: Some(labels),
        host_config: Some(host_config),
        ..Default::default()
    };

    let create_opts = CreateContainerOptions {
        name: container_name.clone(),
        platform: None,
    };
    let created = state
        .docker()
        .create_container(Some(create_opts), config)
        .await
        .context("docker create")?;
    state
        .docker()
        .start_container(&created.id, None::<StartContainerOptions<String>>)
        .await
        .context("docker start")?;

    Ok(TerminalHandle {
        id,
        name,
        container_id: created.id,
        image: state.image().to_string(),
        status: "running".to_string(),
        created_at: chrono_ts_millis(),
    })
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

    // Wipe the scratch volume. Best-effort — if the directory was
    // already cleaned up externally, that's fine.
    let _ = std::fs::remove_dir_all(scratch_path(id)?);
    Ok(())
}

pub async fn find_container_id(state: &AppState, id: &str) -> Result<String> {
    let mut filters = HashMap::new();
    filters.insert(
        "label".to_string(),
        vec![format!("heiervang.ht-termd.id={id}")],
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
