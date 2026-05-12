/**
 * Browser-side video decomposition for models that don't accept video natively.
 *
 * Splits a video File/Blob into two streams the model CAN read:
 * - an evenly-spaced sequence of still frames (JPEG data URLs), for vision
 *   models, and
 * - a WAV-encoded audio track, for audio models.
 *
 * All decoding stays in the browser — no ffmpeg.wasm dependency. That keeps
 * the bundle small but limits us to the codecs the host webview supports
 * (mp4/H264, webm/VP8-9, mov usually). Graceful error messages cover the
 * common failure modes (codec unsupported, audio track missing, seek stall).
 */

export interface VideoDecomposeFrame {
	/** `data:image/jpeg;base64,…` encoded still. */
	dataUrl: string;
	/** Timestamp in the source video this frame was sampled from. */
	timestampSec: number;
}

export interface VideoDecomposeResult {
	frames: VideoDecomposeFrame[];
	/** Null when the video had no decodable audio track. */
	audio: Blob | null;
	durationSec: number;
	widthPx: number;
	heightPx: number;
}

export interface ExtractFramesOptions {
	/** How many still frames to pull; defaults to 8. */
	frameCount?: number;
	/** JPEG quality 0..1; defaults to 0.8. */
	jpegQuality?: number;
	/** Longest-edge cap in pixels; frames larger than this get scaled down so
	 *  we don't ship megabyte JPEGs to the model for no gain. */
	maxDimension?: number;
}

const DEFAULT_FRAME_COUNT = 8;
const DEFAULT_JPEG_QUALITY = 0.8;
const DEFAULT_MAX_DIMENSION = 768;

/**
 * Extract an evenly-spaced set of still frames from a video file.
 *
 * Strategy: seek an off-screen <video> element to each target timestamp,
 * wait for `seeked` + `canplay`, draw to a canvas, toDataURL as JPEG.
 * Works for mp4/webm/mov in Chromium + WebKit.
 */
export async function extractFrames(
	file: File | Blob,
	opts: ExtractFramesOptions = {}
): Promise<{
	frames: VideoDecomposeFrame[];
	durationSec: number;
	widthPx: number;
	heightPx: number;
}> {
	const frameCount = Math.max(1, opts.frameCount ?? DEFAULT_FRAME_COUNT);
	const jpegQuality = opts.jpegQuality ?? DEFAULT_JPEG_QUALITY;
	const maxDim = opts.maxDimension ?? DEFAULT_MAX_DIMENSION;

	const url = URL.createObjectURL(file);
	const video = document.createElement('video');
	video.preload = 'auto';
	video.muted = true;
	video.playsInline = true;
	video.crossOrigin = 'anonymous';
	video.src = url;

	try {
		await new Promise<void>((resolve, reject) => {
			const onLoaded = () => {
				cleanup();
				resolve();
			};
			const onError = () => {
				cleanup();
				reject(new Error('Could not decode video metadata — unsupported codec?'));
			};
			const cleanup = () => {
				video.removeEventListener('loadedmetadata', onLoaded);
				video.removeEventListener('error', onError);
			};
			video.addEventListener('loadedmetadata', onLoaded, { once: true });
			video.addEventListener('error', onError, { once: true });
		});

		const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
		if (!duration) {
			throw new Error('Video reports zero or unknown duration');
		}

		const timestamps = planFrameTimestamps(duration, frameCount);

		const srcW = video.videoWidth;
		const srcH = video.videoHeight;
		const scale = Math.min(1, maxDim / Math.max(srcW, srcH));
		const canvasW = Math.max(1, Math.round(srcW * scale));
		const canvasH = Math.max(1, Math.round(srcH * scale));

		const canvas = document.createElement('canvas');
		canvas.width = canvasW;
		canvas.height = canvasH;
		const ctx = canvas.getContext('2d');
		if (!ctx) throw new Error('Could not acquire 2D canvas context');

		const frames: VideoDecomposeFrame[] = [];
		for (const t of timestamps) {
			await seekTo(video, t);
			ctx.drawImage(video, 0, 0, canvasW, canvasH);
			const dataUrl = canvas.toDataURL('image/jpeg', jpegQuality);
			frames.push({ dataUrl, timestampSec: t });
		}

		return { frames, durationSec: duration, widthPx: srcW, heightPx: srcH };
	} finally {
		video.removeAttribute('src');
		video.load();
		URL.revokeObjectURL(url);
	}
}

function planFrameTimestamps(duration: number, count: number): number[] {
	if (count === 1) return [duration / 2];
	// Evenly spaced, skewed slightly inward so the very first/last frame (often
	// black) aren't the representatives.
	const pad = Math.min(0.25, duration * 0.02);
	const start = pad;
	const end = Math.max(pad, duration - pad);
	const span = end - start;
	return Array.from({ length: count }, (_, i) => start + (span * i) / (count - 1));
}

function seekTo(video: HTMLVideoElement, timestamp: number): Promise<void> {
	return new Promise<void>((resolve, reject) => {
		const onSeeked = () => {
			cleanup();
			resolve();
		};
		const onError = () => {
			cleanup();
			reject(new Error(`Seek to ${timestamp.toFixed(2)}s failed`));
		};
		const cleanup = () => {
			video.removeEventListener('seeked', onSeeked);
			video.removeEventListener('error', onError);
			clearTimeout(timer);
		};
		// A stall guard — some codecs stutter on random seeks. Fail loud after
		// 10s so the overall extraction never hangs the UI indefinitely.
		const timer = setTimeout(() => {
			cleanup();
			reject(new Error(`Seek to ${timestamp.toFixed(2)}s timed out after 10s`));
		}, 10_000);
		video.addEventListener('seeked', onSeeked, { once: true });
		video.addEventListener('error', onError, { once: true });
		video.currentTime = Math.min(video.duration, Math.max(0, timestamp));
	});
}

/**
 * Extract the audio track from a video file and return it as a WAV blob.
 *
 * Uses the standard AudioContext.decodeAudioData path — works for mp4/webm
 * in every webview we target. Returns null if the file has no decodable
 * audio track (silent videos, or codecs the host can't demux).
 */
export async function extractAudio(file: File | Blob): Promise<Blob | null> {
	const buffer = await file.arrayBuffer();
	const AudioCtx =
		(window as unknown as { AudioContext?: typeof AudioContext }).AudioContext ??
		(window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
	if (!AudioCtx) return null;

	// OfflineAudioContext would be faster (no real-time playback needed), but
	// it requires knowing sample rate + duration up front. The regular ctx
	// is cheap — we discard it immediately after decode.
	const ctx = new AudioCtx();
	let audioBuffer: AudioBuffer;
	try {
		// Some hosts mutate the buffer during decode; pass a copy.
		audioBuffer = await ctx.decodeAudioData(buffer.slice(0));
	} catch {
		await ctx.close().catch(() => {});
		return null;
	}
	await ctx.close().catch(() => {});

	return audioBufferToWav(audioBuffer);
}

/**
 * Encode an AudioBuffer to a 16-bit PCM WAV Blob. Minimal implementation;
 * good enough for model ingestion (not for archival audio quality).
 */
function audioBufferToWav(buffer: AudioBuffer): Blob {
	const numChannels = Math.min(buffer.numberOfChannels, 2);
	const sampleRate = buffer.sampleRate;
	const format = 1; // PCM
	const bitDepth = 16;
	const bytesPerSample = bitDepth / 8;
	const blockAlign = numChannels * bytesPerSample;
	const byteRate = sampleRate * blockAlign;
	const dataLength = buffer.length * blockAlign;
	const headerLength = 44;

	const out = new ArrayBuffer(headerLength + dataLength);
	const view = new DataView(out);

	writeString(view, 0, 'RIFF');
	view.setUint32(4, 36 + dataLength, true);
	writeString(view, 8, 'WAVE');
	writeString(view, 12, 'fmt ');
	view.setUint32(16, 16, true);
	view.setUint16(20, format, true);
	view.setUint16(22, numChannels, true);
	view.setUint32(24, sampleRate, true);
	view.setUint32(28, byteRate, true);
	view.setUint16(32, blockAlign, true);
	view.setUint16(34, bitDepth, true);
	writeString(view, 36, 'data');
	view.setUint32(40, dataLength, true);

	const channels: Float32Array[] = [];
	for (let ch = 0; ch < numChannels; ch++) {
		channels.push(buffer.getChannelData(ch));
	}

	let offset = headerLength;
	for (let i = 0; i < buffer.length; i++) {
		for (let ch = 0; ch < numChannels; ch++) {
			const sample = Math.max(-1, Math.min(1, channels[ch][i]));
			view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
			offset += 2;
		}
	}

	return new Blob([out], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, str: string): void {
	for (let i = 0; i < str.length; i++) {
		view.setUint8(offset + i, str.charCodeAt(i));
	}
}

/**
 * Top-level convenience: returns frames + audio in one call.
 */
export async function decomposeVideo(
	file: File | Blob,
	opts: ExtractFramesOptions = {}
): Promise<VideoDecomposeResult> {
	const [framesResult, audio] = await Promise.all([
		extractFrames(file, opts),
		extractAudio(file).catch(() => null)
	]);
	return {
		frames: framesResult.frames,
		audio,
		durationSec: framesResult.durationSec,
		widthPx: framesResult.widthPx,
		heightPx: framesResult.heightPx
	};
}
