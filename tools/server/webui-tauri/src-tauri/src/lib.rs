use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, RunEvent, WindowEvent};

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
	// Force webkit2gtk's media pipeline onto the legacy `playbin` rather
	// than `playbin3`. On Arch with GStreamer 1.28 + webkit2gtk-4.1
	// 2.50, playbin3's caps registration trips a
	// `gst_value_collect_int_range` assertion when probing some
	// fragmented MP4s and the WebProcess crashes — taking down the
	// entire shell. The fallback path is slightly less efficient but
	// stable. Setting this in the Rust entry-point (rather than only
	// in the .desktop launcher) means a `cargo run` / direct binary
	// invocation also picks up the workaround.
	#[cfg(target_os = "linux")]
	if std::env::var_os("WEBKIT_GST_USE_PLAYBIN3").is_none() {
		std::env::set_var("WEBKIT_GST_USE_PLAYBIN3", "0");
	}

	// Defuse webkit2gtk's DMABUF renderer + accelerated compositing under
	// Wayland on Arch. WPEBackend-fdo / DMABUF combine badly with several
	// driver/compositor stacks (NVIDIA proprietary, but also AMD on
	// wlroots-based compositors like Hyprland 0.5x), producing
	// "failed to create GBM buffer" errors and intermittent WebProcess
	// crashes on launch or resize (tauri-apps/tauri#9394, #13493). The
	// .desktop launchers already set these, but a `cargo run` / direct
	// binary invocation skips the launcher — set them in the Rust entry
	// so they always apply.
	#[cfg(target_os = "linux")]
	if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
		std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
	}
	#[cfg(target_os = "linux")]
	if std::env::var_os("WEBKIT_DISABLE_COMPOSITING_MODE").is_none() {
		std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
	}

	// Disable GStreamer hardware video decoders inside webkit2gtk. VA-API
	// and NVDEC paths on Arch with webkit2gtk-4.1 2.50 + GStreamer 1.28
	// trip the WebProcess for several common MP4 codec profiles —
	// especially when `gst-plugins-bad` (the package that ships HEVC /
	// AV1 / many newer codec elements) is missing on the host, which
	// leaves the pipeline negotiating against an incomplete codec set.
	// Software decode via `gst-libav` is slower but stable. Set this
	// alongside the playbin3 + DMABUF guards so the entire video path is
	// pinned to the known-stable software-decode lane.
	#[cfg(target_os = "linux")]
	if std::env::var_os("WEBKIT_GST_USE_HARDWARE_VIDEO_DECODERS").is_none() {
		std::env::set_var("WEBKIT_GST_USE_HARDWARE_VIDEO_DECODERS", "0");
	}
	#[cfg(target_os = "linux")]
	if std::env::var_os("WEBKIT_GST_DISABLE_VAAPI").is_none() {
		std::env::set_var("WEBKIT_GST_DISABLE_VAAPI", "1");
	}

	tauri::Builder::default()
		// Pin the embedded server to 127.0.0.1 — left to its default
		// "localhost" the plugin's tiny-http binds to whichever IP
		// glibc returns first, which on this stack is `::1`. WebKit2GTK
		// then resolves the windows.url's `localhost` to 127.0.0.1
		// first via /etc/hosts and hits ConnRefused, leaving the
		// webview white. Pinning IPv4 on both ends avoids that
		// asymmetry.
		.plugin(tauri_plugin_localhost::Builder::new(RELEASE_PORT).host("127.0.0.1").build())
		// Routes `fetch` calls from JS through reqwest in the Tauri runtime
		// rather than WebKit's resource loader. ht-termd traffic uses this
		// path so a stalled / throttled webview never strands a `send_keys`
		// invocation. Scope is locked down in `capabilities/default.json`.
		.plugin(tauri_plugin_http::init())
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

			// System tray icon + menu. Click on the icon shows/hides the
			// main window; the menu carries Show / Hide / Quit plus a
			// placeholder slot for the backend selector we'll wire up
			// once the Settings → backendBaseUrl plumbing is exposed to
			// the Rust side. Window hide-on-close (Tauri's RunEvent
			// branch below) keeps the app running in the tray when the
			// user closes the main window — Docker Desktop pattern.
			let show_item = MenuItem::with_id(app, "tray-show", "Show heierchat", true, None::<&str>)?;
			let hide_item = MenuItem::with_id(app, "tray-hide", "Hide window", true, None::<&str>)?;
			let separator = PredefinedMenuItem::separator(app)?;
			let backend_item = MenuItem::with_id(
				app,
				"tray-backend",
				"Backend: (configured in Settings)",
				false,
				None::<&str>,
			)?;
			let quit_item = MenuItem::with_id(app, "tray-quit", "Quit heierchat", true, None::<&str>)?;
			let menu = Menu::with_items(
				app,
				&[&show_item, &hide_item, &separator, &backend_item, &separator, &quit_item],
			)?;

			let _tray = TrayIconBuilder::with_id("main-tray")
				.tooltip("heierchat")
				.icon(app.default_window_icon().cloned().unwrap_or_else(|| {
					// Fall back to a transparent 1×1 RGBA so the tray slot still
					// appears even on platforms where the bundled icon is missing.
					tauri::image::Image::new_owned(vec![0u8; 4], 1, 1)
				}))
				.menu(&menu)
				.show_menu_on_left_click(false)
				.on_menu_event(|app, event| match event.id.as_ref() {
					"tray-show" => {
						if let Some(w) = app.get_webview_window("main") {
							let _ = w.show();
							let _ = w.set_focus();
							let _ = w.unminimize();
						}
					}
					"tray-hide" => {
						if let Some(w) = app.get_webview_window("main") {
							let _ = w.hide();
						}
					}
					"tray-quit" => {
						app.exit(0);
					}
					_ => {}
				})
				.on_tray_icon_event(|tray, event| {
					if let TrayIconEvent::Click {
						button: MouseButton::Left,
						button_state: MouseButtonState::Up,
						..
					} = event
					{
						let app = tray.app_handle();
						if let Some(w) = app.get_webview_window("main") {
							if w.is_visible().unwrap_or(false) {
								let _ = w.hide();
							} else {
								let _ = w.show();
								let _ = w.set_focus();
							}
						}
					}
				})
				.build(app)?;

			Ok(())
		})
		.build(tauri::generate_context!())
		.expect("error while building tauri application")
		.run(|app, event| {
			// Close-to-tray: intercept the main window's close request,
			// prevent the actual close, and hide it instead. The user
			// re-opens via the tray icon (left-click toggle) or the
			// Show heierchat menu item. Quit is reachable from the tray
			// menu and from app.exit() programmatically. Docker Desktop
			// pattern: closing the window doesn't quit the app, it just
			// minimises it to the tray.
			if let RunEvent::WindowEvent {
				label,
				event: WindowEvent::CloseRequested { api, .. },
				..
			} = event
			{
				if label == "main" {
					api.prevent_close();
					if let Some(w) = app.get_webview_window("main") {
						let _ = w.hide();
					}
				}
			}
		});
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
