use tauri::Manager;

/// Port the embedded frontend is served on in release builds.
///
/// Locked to 5173 to match `cargo tauri dev`'s vite default. Same port →
/// same origin → same IndexedDB / localStorage / OPFS, so a user who
/// accumulated artifacts and settings under `cargo tauri dev` does not
/// lose them after switching to the optimized release binary. Only one
/// of {vite, the embedded server} can bind at a time, which is the
/// intended invariant — you run dev OR release, not both.
const RELEASE_PORT: u16 = 5173;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
	tauri::Builder::default()
		// Pin the embedded server to 127.0.0.1 — left to its default
		// "localhost" the plugin's tiny-http binds to whichever IP
		// glibc returns first, which on this stack is `::1`. WebKit2GTK
		// then resolves the windows.url's `localhost` to 127.0.0.1
		// first via /etc/hosts and hits ConnRefused, leaving the
		// webview white. Pinning IPv4 on both ends avoids that
		// asymmetry.
		.plugin(tauri_plugin_localhost::Builder::new(RELEASE_PORT).host("127.0.0.1").build())
		.setup(|app| {
			if cfg!(debug_assertions) {
				app.handle().plugin(
					tauri_plugin_log::Builder::default()
						.level(log::LevelFilter::Info)
						.build(),
				)?;
			}

			// Enable mic/camera capture inside the bundled webview. The browser's
			// getUserMedia API needs the webkit2gtk settings flags toggled on and
			// permission_request approved — otherwise the chat mic button returns
			// NotAllowedError even when the desktop app has OS-level mic access.
			#[cfg(target_os = "linux")]
			if let Some(window) = app.get_webview_window("main") {
				if let Err(err) = enable_media_capture_linux(&window) {
					log::warn!("failed to enable webview media capture: {err:?}");
				}
			}

			// Inject platform defaults into the webview's global scope so
			// a fresh install on Android auto-targets the user's
			// tailnet-reachable llama.cpp instead of showing an empty
			// Settings form. Desktop builds with HT_DEFAULT_* unset
			// inject empty strings — the webui treats those as "no
			// preference" and falls back to llama-server's /props as
			// before.
			if let Some(window) = app.get_webview_window("main") {
				let script = defaults_init_script();
				if let Err(err) = window.eval(&script) {
					log::warn!("failed to inject platform defaults: {err:?}");
				}
			}

			// HT_OPEN_DEVTOOLS=1 → pop the inspector on launch. Used to
			// diagnose splash-stuck regressions on release builds where
			// the bundle import fails silently. open_devtools is only
			// linked when the tauri/devtools feature is enabled.
			if std::env::var("HT_OPEN_DEVTOOLS").is_ok() {
				if let Some(window) = app.get_webview_window("main") {
					window.open_devtools();
				}
			}

			Ok(())
		})
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
}

/// Defaults baked into the bundle at build time. Read by the webui as
/// fallbacks when `config().backendBaseUrl` / `terminalsBaseUrl` are
/// empty — the user can still override in Settings.
///
/// These values come from `HT_DEFAULT_BACKEND_URL` /
/// `HT_DEFAULT_TERMINALS_URL` at build time, so a desktop bundle and
/// an Android APK can carry different presets without a code
/// difference. An Android APK is built with the tailnet URLs so the
/// phone just works.
fn defaults_init_script() -> String {
	let backend = option_env!("HT_DEFAULT_BACKEND_URL").unwrap_or("");
	let terminals = option_env!("HT_DEFAULT_TERMINALS_URL").unwrap_or("");
	// Paired with `HT_DEFAULT_TERMINALS_URL`. Required when the
	// remote termd was launched with `--token` (tailnet / LAN
	// deployments). Leave unset for loopback-only bundles.
	let terminals_token = option_env!("HT_DEFAULT_TERMINALS_TOKEN").unwrap_or("");
	format!(
		"window.__HT_DEFAULT_BACKEND_URL__ = {backend_js}; \
		 window.__HT_DEFAULT_TERMINALS_URL__ = {terminals_js}; \
		 window.__HT_DEFAULT_TERMINALS_TOKEN__ = {terminals_token_js};",
		backend_js = serde_json::to_string(backend).unwrap_or_else(|_| "\"\"".to_string()),
		terminals_js = serde_json::to_string(terminals).unwrap_or_else(|_| "\"\"".to_string()),
		terminals_token_js =
			serde_json::to_string(terminals_token).unwrap_or_else(|_| "\"\"".to_string()),
	)
}

#[cfg(target_os = "linux")]
fn enable_media_capture_linux(window: &tauri::WebviewWindow) -> tauri::Result<()> {
	use webkit2gtk::{PermissionRequestExt, SettingsExt, WebViewExt};

	window.with_webview(|webview| {
		let wv = webview.inner();
		if let Some(settings) = WebViewExt::settings(&wv) {
			settings.set_enable_media_stream(true);
			settings.set_enable_mediasource(true);
			settings.set_media_playback_requires_user_gesture(false);
		}
		wv.connect_permission_request(|_wv, request: &webkit2gtk::PermissionRequest| {
			request.allow();
			true
		});
	})
}
