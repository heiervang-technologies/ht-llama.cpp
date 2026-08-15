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
//! We deliberately inspect `iptables -S DOCKER-USER` instead of
//! reading the netlink table directly so the check mirrors what
//! `unleash sandbox status` installs.

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

pub async fn sandbox_status(docker: &Docker, network: &str, image: &str) -> Result<SandboxStatus> {
    let docker_ok = docker.ping().await.is_ok();
    let info = if docker_ok {
        docker.info().await.ok()
    } else {
        None
    };

    let runsc_ok = info
        .as_ref()
        .and_then(|i| i.runtimes.as_ref())
        .map(|rt| rt.contains_key("runsc"))
        .unwrap_or(false);

    let sandbox_subnets = network_subnets(docker, network).await.ok();
    let network_ok = sandbox_subnets.is_some();
    let image_ok = image_exists(docker, image).await.unwrap_or(false);
    let iptables_ok = match sandbox_subnets {
        Some(subnets) => check_iptables(&subnets).await,
        None => IpTablesStatus::Missing,
    };

    Ok(SandboxStatus {
        docker_ok,
        runsc_ok,
        network_ok,
        iptables_ok,
        image_ok,
    })
}

pub async fn assert_sandbox_ready(docker: &Docker, network: &str, image: &str) -> Result<()> {
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
            return Err(anyhow!(
                "iptables LAN-drop rules cannot be verified — grant passwordless read access to `iptables -S DOCKER-USER` or run ht-termd with sufficient privileges"
            ));
        }
    }
    if !s.image_ok {
        return Err(anyhow!(
            "Container image `{image}` not found — run `unleash sandbox setup` to build it"
        ));
    }
    Ok(())
}

async fn network_subnets(docker: &Docker, network: &str) -> Result<Vec<String>> {
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
    if !icc_off {
        return Err(anyhow!(
            "sandbox network has inter-container communication enabled"
        ));
    }

    let subnets = net
        .ipam
        .and_then(|ipam| ipam.config)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|config| config.subnet)
        .filter(|subnet| !subnet.contains(':'))
        .collect::<Vec<_>>();
    if subnets.is_empty() {
        return Err(anyhow!("sandbox network has no IPv4 subnet"));
    }
    Ok(subnets)
}

async fn image_exists(docker: &Docker, image: &str) -> Result<bool> {
    Ok(docker.inspect_image(image).await.is_ok())
}

/// Check that DOCKER-USER has a source-scoped DROP rule for each
/// required private range and each IPv4 subnet of the sandbox network.
async fn check_iptables(sandbox_subnets: &[String]) -> IpTablesStatus {
    // iptables can only be read as root. Try direct first (unlikely
    // to succeed under the user daemon), then fall back to
    // `sudo -n` which works when a passwordless sudoers entry exists
    // for iptables. If neither path reads the table, return
    // `Unknown`. Container creation treats that as a hard refusal.
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
    iptables_status_from_dump(&String::from_utf8_lossy(&output), sandbox_subnets)
}

fn iptables_status_from_dump(dump: &str, sandbox_subnets: &[String]) -> IpTablesStatus {
    const LAN_RANGES: [&str; 4] = [
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "169.254.0.0/16",
    ];

    for subnet in sandbox_subnets {
        for range in LAN_RANGES {
            let found = dump.lines().any(|line| {
                let fields = line.split_whitespace().collect::<Vec<_>>();
                option_value(&fields, "-s") == Some(subnet.as_str())
                    && option_value(&fields, "-d") == Some(range)
                    && option_value(&fields, "-j") == Some("DROP")
            });
            if !found {
                return IpTablesStatus::Missing;
            }
        }
    }
    IpTablesStatus::Ok
}

fn option_value<'a>(fields: &'a [&str], option: &str) -> Option<&'a str> {
    fields
        .windows(2)
        .find_map(|pair| (pair[0] == option).then_some(pair[1]))
}

#[cfg(test)]
mod tests {
    use super::{iptables_status_from_dump, IpTablesStatus};

    const SUBNET: &str = "172.30.0.0/16";
    const COMPLETE: &str = "\
-A DOCKER-USER -s 172.30.0.0/16 -d 10.0.0.0/8 -j DROP\n\
-A DOCKER-USER -s 172.30.0.0/16 -d 172.16.0.0/12 -j DROP\n\
-A DOCKER-USER -s 172.30.0.0/16 -d 192.168.0.0/16 -j DROP\n\
-A DOCKER-USER -s 172.30.0.0/16 -d 169.254.0.0/16 -j DROP\n";

    #[test]
    fn accepts_complete_source_scoped_rules() {
        assert_eq!(
            iptables_status_from_dump(COMPLETE, &[SUBNET.to_string()]),
            IpTablesStatus::Ok
        );
    }

    #[test]
    fn rejects_missing_destination() {
        assert_eq!(
            iptables_status_from_dump(
                &COMPLETE.replace("-d 169.254.0.0/16", "-d 100.64.0.0/10"),
                &[SUBNET.to_string()]
            ),
            IpTablesStatus::Missing
        );
    }

    #[test]
    fn rejects_rules_for_another_source() {
        assert_eq!(
            iptables_status_from_dump(COMPLETE, &["172.31.0.0/16".to_string()]),
            IpTablesStatus::Missing
        );
    }
}
