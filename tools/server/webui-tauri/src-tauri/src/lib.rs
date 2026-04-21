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

			Ok(())
		})
		.run(tauri::generate_context!())
		.expect("error while running tauri application");
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
