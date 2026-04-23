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
    if s.iptables_ok != IpTablesStatus::Ok {
        return Err(anyhow!(
            "iptables LAN-drop rules missing or unverifiable — run `sudo unleash sandbox setup`"
        ));
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

/// Check each required DROP rule via `iptables -C DOCKER-USER …`.
/// Exit code 0 = rule exists; anything else (including "command not
/// found" or "permission denied") trips the Unknown branch.
async fn check_iptables() -> IpTablesStatus {
    let subnet_ranges = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
    ];
    // We can't easily know the sandbox subnet (172.30.0.0/16 by
    // convention) without reading the compose/shell scripts. The
    // rule shape written by unleash's setup script is
    //   -s 172.30.0.0/16 -d <range> -j DROP
    // We therefore check for rules that DROP to each `<range>`
    // regardless of source — any presence of a DROP rule with that
    // destination counts as covered.
    let mut all_ok = true;
    let mut any_ran = false;
    for range in subnet_ranges {
        let out = Command::new("iptables")
            .args(["-C", "DOCKER-USER", "-d", range, "-j", "DROP"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await;
        match out {
            Ok(status) => {
                any_ran = true;
                if !status.success() {
                    all_ok = false;
                }
            }
            Err(_) => return IpTablesStatus::Unknown,
        }
    }
    if !any_ran {
        return IpTablesStatus::Unknown;
    }
    if all_ok {
        IpTablesStatus::Ok
    } else {
        IpTablesStatus::Missing
    }
}
