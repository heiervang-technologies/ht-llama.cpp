use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
	tauri::Builder::default()
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
