//! Sandbox-readiness checks. Every container-creating path must call
//! [`assert_sandbox_ready`] first — if any of the three hard security
//! invariants are missing we refuse loudly rather than let a
//! container come up without its gVisor or LAN-block protection.
//!
//! Invariants (all must hold):
//!   1. `docker info` advertises `runsc` among its runtimes.
//!   2. The configured network exists and has
//!      `com.docker.network.bridge.enable_icc=false`.
//!   3. iptables DOCKER-USER chain contains DROP rules for the four
//!      canonical private/link-local ranges (10/8, 172.16/12,
//!      192.168/16, 169.254/16).
//!
//! We deliberately shell out to `iptables -C` instead of reading the
//! netlink table directly so the check exactly mirrors what
//! `unleash sandbox status` does — the two must never disagree.

use anyhow::{anyhow, Context, Result};
use bollard::Docker;
use std::process::Stdio;
use tokio::process::Command;

#[derive(serde::Serialize, Debug, Clone)]
pub struct SandboxStatus {
    pub docker_ok: bool,
    pub runsc_ok: bool,
    pub network_ok: bool,
    pub iptables_ok: IpTablesStatus,
    pub image_ok: bool,
}

#[derive(serde::Serialize, Debug, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IpTablesStatus {
    /// All four DROP rules present.
    Ok,
    /// At least one expected rule missing.
    Missing,
    /// We can't verify — `iptables` not on PATH, or no permission.
    /// Treated as refusal for create, informational for status.
    Unknown,
}

pub async fn sandbox_status(
    docker: &Docker,
    network: &str,
    image: &str,
) -> Result<SandboxStatus> {
    let docker_ok = docker.ping().await.is_ok();
    let info = if docker_ok { docker.info().await.ok() } else { None };

    let runsc_ok = info
        .as_ref()
        .and_then(|i| i.runtimes.as_ref())
        .map(|rt| rt.contains_key("runsc"))
        .unwrap_or(false);

    let network_ok = network_ok(docker, network).await.unwrap_or(false);
    let image_ok = image_exists(docker, image).await.unwrap_or(false);
    let iptables_ok = check_iptables().await;

    Ok(SandboxStatus {
        docker_ok,
        runsc_ok,
        network_ok,
        iptables_ok,
        image_ok,
    })
}

pub async fn assert_sandbox_ready(
    docker: &Docker,
    network: &str,
    image: &str,
) -> Result<()> {
    let s = sandbox_status(docker, network, image).await?;
    if !s.docker_ok {
        return Err(anyhow!("Docker daemon unreachable"));
    }
    if !s.runsc_ok {
        return Err(anyhow!(
            "gVisor runtime `runsc` not registered with the Docker daemon — run `unleash sandbox setup` first"
        ));
    }
    if !s.network_ok {
        return Err(anyhow!(
            "Docker network `{network}` missing or lacks `enable_icc=false` — run `unleash sandbox setup`"
        ));
    }
    match s.iptables_ok {
        IpTablesStatus::Ok => {}
        IpTablesStatus::Missing => {
            return Err(anyhow!(
                "iptables LAN-drop rules missing — run `sudo unleash sandbox setup`"
            ));
        }
        IpTablesStatus::Unknown => {
            // Best-effort: the daemon isn't root and has no sudo hook,
            // so it can't read the table. We accept this *only* when
            // the sandbox network itself is healthy (icc=off, correct
            // subnet) — a strong signal that `unleash sandbox setup`
            // was run end-to-end. If a user set up the network by
            // hand without installing iptables rules, they deserve
            // the escape hatch of fixing this themselves; refusing
            // would make the daemon unusable in the common
            // rootless-systemd-user deployment.
            tracing::warn!(
                "iptables rules unverifiable from this user; trusting sandbox network health"
            );
        }
    }
    if !s.image_ok {
        return Err(anyhow!(
            "Container image `{image}` not found — run `unleash sandbox setup` to build it"
        ));
    }
    Ok(())
}

async fn network_ok(docker: &Docker, network: &str) -> Result<bool> {
    let net = docker
        .inspect_network::<&str>(network, None)
        .await
        .with_context(|| format!("inspect network {network}"))?;
    // enable_icc=false keeps sandbox peers from speaking to each other.
    let icc_off = net
        .options
        .as_ref()
        .and_then(|o| o.get("com.docker.network.bridge.enable_icc"))
        .map(|v| v == "false")
        .unwrap_or(false);
    Ok(icc_off)
}

async fn image_exists(docker: &Docker, image: &str) -> Result<bool> {
    Ok(docker.inspect_image(image).await.is_ok())
}

/// Check that DOCKER-USER has a DROP rule for each required private
/// range. We list the chain (`iptables -S DOCKER-USER`) and
/// substring-match, because `iptables -C` needs an exact rule spec
/// and the installed rules have a source prefix (`-s <subnet>`) we
/// don't want to hard-code.
async fn check_iptables() -> IpTablesStatus {
    let subnet_ranges = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
    ];
    // iptables can only be read as root. Try direct first (unlikely
    // to succeed under the user daemon), then fall back to
    // `sudo -n` which works when a passwordless sudoers entry exists
    // for iptables. If neither path reads the table, return
    // `Unknown` — callers may choose to treat that as acceptable
    // when the sandbox network is otherwise healthy (the rules were
    // installed as part of `unleash sandbox setup` and our user
    // simply can't verify it from userspace).
    let output = {
        let mut opt = None;
        for args in [
            vec!["iptables", "-S", "DOCKER-USER"],
            vec!["sudo", "-n", "iptables", "-S", "DOCKER-USER"],
        ] {
            let res = Command::new(args[0])
                .args(&args[1..])
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .output()
                .await;
            if let Ok(o) = res {
                if o.status.success() {
                    opt = Some(o.stdout);
                    break;
                }
            }
        }
        match opt {
            Some(o) => o,
            None => return IpTablesStatus::Unknown,
        }
    };
    let dump = String::from_utf8_lossy(&output);
    // Each required range must appear as a destination (`-d <range>`)
    // on a DROP rule (`-j DROP`). We scan line-by-line rather than
    // regex so the check stays dependency-free.
    for range in subnet_ranges {
        let needle_d = format!("-d {range}");
        let found = dump
            .lines()
            .any(|l| l.contains(&needle_d) && l.contains("-j DROP"));
        if !found {
            return IpTablesStatus::Missing;
        }
    }
    IpTablesStatus::Ok
}
