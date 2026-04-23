import argparse
import os

import requests


def parse_args():
    parser = argparse.ArgumentParser(description="Offline ASR CLI client for REST API.")
    parser.add_argument("--input-file", "-i", required=True, help="Input audio/video file path.")
    parser.add_argument(
        "--api-url",
        "-u",
        default="http://127.0.0.1:18000/api/v1/offline/transcribe",
        help="Offline transcription API URL.",
    )
    parser.add_argument("--context", "-c", default="", help="Prompt context for ASR.")
    parser.add_argument(
        "--use-forced-aligner",
        action="store_true",
        help="Request forced alignment with Qwen3-ForcedAligner-0.6B when available.",
    )
    parser.add_argument(
        "--save-text",
        action="store_true",
        help="Save transcription text to `<input_basename>.txt`.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

    with open(args.input_file, "rb") as f:
        files = {"audio_file": (os.path.basename(args.input_file), f, "application/octet-stream")}
        data = {
            "context": args.context,
            "use_forced_aligner": str(bool(args.use_forced_aligner)).lower(),
        }
        response = requests.post(args.api_url, files=files, data=data, timeout=600)

    response.raise_for_status()
    result = response.json()

    print("=== Offline Transcription ===")
    print(f"Language: {result.get('language', '')}")
    print(f"Segments: {result.get('segment_count', 0)}")
    print("Text:")
    print(result.get("text", ""))

    forced_aligner = result.get("forced_aligner", {})
    if forced_aligner.get("requested"):
        print(
            f"Forced Aligner Available: {forced_aligner.get('available', False)} | "
            f"Message: {forced_aligner.get('message', '')}"
        )

    if args.save_text:
        save_file = os.path.splitext(args.input_file)[0] + ".txt"
        with open(save_file, "w", encoding="utf-8") as f:
            f.write((result.get("language") or "") + "\n")
            f.write((result.get("text") or "") + "\n")
        print(f"Saved transcript to: {save_file}")


if __name__ == "__main__":
    main()
