#!/usr/bin/env python3
"""Play streamed PCM16 audio from audio.cpp's /v1/audio/speech endpoint.

Install: python -m pip install requests sounddevice

Example (Qwen3-TTS, server model configured with mode='streaming'):
  python audiocpp_tts_stream_player.py \
      --model qwen3-tts-stream \
      --text 'こんにちは。これはストリーミング音声合成のテストです。' \
      --sample-rate 24000 \
      --options '{"stream_frames_per_event":4}'

The server must be running. PCM16, mono, little-endian is assumed by default.
Use --transport audio for raw PCM instead of SSE events.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import queue
import sys
import threading
import time
from typing import Iterator

import requests


class PCMPlayer:
    """Non-blocking producer queue + dedicated, blocking PortAudio playback thread."""

    def __init__(
        self,
        sample_rate: int,
        channels: int = 1,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.frame_bytes = 2 * channels  # signed 16-bit PCM, interleaved
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._playback_loop, name="pcm-playback", daemon=True
        )
        self.error: Exception | None = None
        self.chunks = 0
        self.audio_bytes = 0

    def start(self) -> None:
        self._thread.start()

    def enqueue(self, pcm: bytes) -> None:
        if self.error is not None:
            raise RuntimeError("Audio playback failed") from self.error
        if not pcm:
            return
        if len(pcm) % self.frame_bytes:
            raise ValueError(
                f"PCM chunk has {len(pcm)} bytes; "
                f"must be divisible by {self.frame_bytes} (16-bit x channels)"
            )
        # The HTTP receiving thread never waits for speaker playback here.
        self._queue.put_nowait(pcm)
        self.chunks += 1
        self.audio_bytes += len(pcm)

    def finish(self) -> None:
        self._queue.put_nowait(None)
        self._thread.join()
        if self.error is not None:
            raise RuntimeError(f"Audio output error: {self.error}") from self.error

    def _playback_loop(self) -> None:
        try:
            # Do not open the device until there is something to play.
            first = self._queue.get()
            if first is None:
                return

            # Import on the playback thread to keep parsing/test helpers reusable.
            import sounddevice as sd

            with sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                device=self.device,
                blocksize=0,
                latency="low",
            ) as stream:
                current = first
                while current is not None:
                    # Blocking write only on THIS thread; HTTP receiver can keep
                    # enqueueing chunks while previous chunks are playing.
                    stream.write(current)
                    current = self._queue.get()
            # Exiting RawOutputStream normally stops/drains buffered playback.
        except Exception as exc:
            self.error = exc


def iter_sse_data(response: requests.Response) -> Iterator[str]:
    """Yield complete SSE data payloads, never assuming one TCP read = one event."""
    lines: list[bytes] = []
    # For chunked HTTP, chunk_size=None follows received chunks without
    # holding a short first audio event for the default 512-byte buffer.
    for raw_line in response.iter_lines(chunk_size=None, decode_unicode=False):
        line = raw_line.rstrip(b"\r")
        if line == b"":
            if lines:
                yield b"\n".join(lines).decode("utf-8")
                lines.clear()
        elif line.startswith(b"data:"):
            lines.append(line[5:].lstrip(b" "))
        # Ignore SSE comments and optional event:/id:/retry: fields.
    if lines:
        yield b"\n".join(lines).decode("utf-8")


def receive_sse(response: requests.Response, player: PCMPlayer) -> None:
    saw_audio_done = False
    for data in iter_sse_data(response):
        if data == "[DONE]":
            break

        event = json.loads(data)
        event_type = event.get("type")

        if event_type == "speech.audio.delta":
            encoded = event.get("audio")
            if not isinstance(encoded, str):
                raise ValueError("speech.audio.delta is missing its audio field")
            try:
                pcm = base64.b64decode(encoded, validate=True)
            except binascii.Error as exc:
                raise ValueError("Invalid base64 PCM from server") from exc
            player.enqueue(pcm)

        elif event_type == "speech.audio.done":
            saw_audio_done = True
            if "timing" in event:
                print(f"[server timing] {json.dumps(event['timing'], ensure_ascii=False)}")

        elif event_type == "error":
            error = event.get("error", event)
            if isinstance(error, dict):
                message = error.get("message", str(error))
            else:
                message = str(error)
            raise RuntimeError(f"Server error: {message}")

    if not saw_audio_done:
        raise RuntimeError("SSE terminated without speech.audio.done")


def receive_raw_pcm(response: requests.Response, player: PCMPlayer) -> None:
    """Raw HTTP chunk boundaries need not correspond to PCM sample boundaries."""
    remainder = b""
    frame_bytes = player.frame_bytes
    for data in response.iter_content(chunk_size=None):
        if not data:
            continue
        data = remainder + data
        full_bytes = (len(data) // frame_bytes) * frame_bytes
        if full_bytes:
            player.enqueue(data[:full_bytes])
        remainder = data[full_bytes:]
    if remainder:
        raise ValueError(f"Raw PCM stream ended with {len(remainder)} incomplete bytes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receive streaming PCM from audiocpp_server and play immediately."
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8080/v1/audio/speech",
        help="audio.cpp TTS endpoint URL",
    )
    parser.add_argument("--model", required=True, help="Configured server model ID")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--transport", choices=("sse", "audio"), default="sse",
        help="SSE base64 PCM or raw binary PCM (default: sse)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=24000,
        help="Actual PCM rate from the model, e.g. 24000 for Qwen3-TTS",
    )
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument(
        "--device", default=None,
        help="PortAudio output device index or name (default: system default)",
    )
    parser.add_argument(
        "--voice-ref", default=None,
        help="Optional server-accessible reference WAV path for cloning",
    )
    parser.add_argument("--reference-text", default=None)
    parser.add_argument(
        "--options", default="{}",
        help='Model-specific request options as JSON, e.g. \'{"stream_frames_per_event":4}\'',
    )
    parser.add_argument(
        "--read-timeout", type=float, default=300.0,
        help="Seconds of network inactivity before failing (default: 300)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_rate <= 0 or args.channels <= 0:
        raise ValueError("--sample-rate and --channels must be positive")

    options = json.loads(args.options)
    if not isinstance(options, dict):
        raise ValueError("--options must be a JSON object")

    payload: dict[str, object] = {
        "model": args.model,
        "input": args.text,
        "response_format": "pcm",
        "stream_format": args.transport,
    }
    if options:
        payload["options"] = options
    if args.voice_ref:
        payload["voice_ref"] = args.voice_ref
    if args.reference_text:
        payload["reference_text"] = args.reference_text

    device: int | str | None = args.device
    if isinstance(device, str) and device.isdecimal():
        device = int(device)

    player = PCMPlayer(
        sample_rate=args.sample_rate,
        channels=args.channels,
        device=device,
    )
    player.start()
    started = time.perf_counter()
    error: BaseException | None = None
    try:
        with requests.post(
            args.url,
            json=payload,
            stream=True,
            headers={
                "Accept": (
                    "text/event-stream" if args.transport == "sse"
                    else "application/octet-stream"
                ),
                "Cache-Control": "no-cache",
            },
            timeout=(10.0, args.read_timeout),
        ) as response:
            if response.status_code >= 400:
                # Server-side exceptions can carry the actual cause in a JSON body.
                # Do not let raise_for_status() hide that useful diagnostic.
                details = response.text.strip()
                if not details:
                    details = "(empty response body; check audiocpp_server logs)"
                raise RuntimeError(
                    f"HTTP {response.status_code} {response.reason} from {args.url}\n"
                    f"Server response: {details[:8192]}"
                )
            content_type = response.headers.get("Content-Type", "").lower()
            if args.transport == "sse" and "text/event-stream" not in content_type:
                raise RuntimeError(f"Expected text/event-stream, got {content_type!r}")
            if args.transport == "audio" and "text/event-stream" in content_type:
                raise RuntimeError("Server returned SSE rather than raw PCM")

            print(f"[connected] {args.model} / {args.transport}", flush=True)
            if args.transport == "sse":
                receive_sse(response, player)
            else:
                receive_raw_pcm(response, player)
    except BaseException as exc:
        error = exc
    finally:
        # Even on network completion, do not exit until queued audio is played.
        if player.chunks:
            print(f"[playback] draining {player.chunks} queued PCM chunk(s) ...", flush=True)
        else:
            print("[playback] no PCM chunks received", flush=True)
        try:
            player.finish()
        except Exception as exc:
            if error is None:
                error = exc

    if error is not None:
        raise error

    duration = player.audio_bytes / (args.sample_rate * args.channels * 2)
    print(
        f"[done] {player.chunks} PCM chunks, "
        f"{duration:.2f}s audio, {time.perf_counter() - started:.2f}s wall time"
    )
    if player.chunks == 0:
        raise RuntimeError("No audio chunks received")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
