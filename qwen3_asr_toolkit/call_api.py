import argparse
import os
from datetime import timedelta
from urllib.parse import urlparse

import requests
import srt
from openai import OpenAI

from qwen3_asr_toolkit.env_utils import load_project_dotenv
from qwen3_asr_toolkit.offline_transcriber import OfflineTranscriber
from qwen3_asr_toolkit.qwen3asr import QwenASR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Local offline transcription via vLLM OpenAI-compatible API with VAD <= 60s per chunk."
    )
    parser.add_argument("--input-file", "-i", type=str, required=True, help="Input media file path or URL.")
    parser.add_argument("--context", "-c", type=str, default="", help="Context prompt for ASR.")
    parser.add_argument("--api-key", "-key", type=str, help="API key for the vLLM server.")
    parser.add_argument(
        "--base-url",
        "-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="Base URL for the OpenAI-compatible vLLM server.",
    )
    parser.add_argument("--model", "-m", type=str, default=None, help="Model name on the vLLM server.")
    parser.add_argument("--num-threads", "-j", type=int, default=4, help="Parallel workers.")
    parser.add_argument(
        "--vad-target-segment-s",
        type=int,
        default=45,
        help="Target VAD chunk length in seconds.",
    )
    parser.add_argument(
        "--vad-max-segment-s",
        type=int,
        default=60,
        help="Hard max chunk length after VAD; no chunk will exceed this.",
    )
    parser.add_argument(
        "--use-forced-aligner",
        action="store_true",
        help="Request Qwen3-ForcedAligner-0.6B timestamps (currently optional and may be unavailable).",
    )
    parser.add_argument("--save-srt", "-srt", action="store_true", help="Save SRT subtitle file.")
    parser.add_argument("--silence", "-s", action="store_true", help="Silence mode.")
    return parser.parse_args()


def resolve_model(api_key: str, base_url: str, silence: bool = False) -> str:
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No model found on vLLM server; please set --model explicitly.")
    model = models.data[0].id
    if not silence:
        print(f"Auto-detected model: {model}")
    return model


def main():
    load_project_dotenv()
    args = parse_args()
    input_file = args.input_file

    if input_file.startswith(("http://", "https://")):
        response = requests.head(input_file, allow_redirects=True, timeout=10)
        if response.status_code >= 400:
            raise FileNotFoundError(f"HTTP input is inaccessible: {input_file} ({response.status_code})")
    elif not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    model = args.model or resolve_model(args.api_key, args.base_url, silence=args.silence)
    asr_client = QwenASR(model=model, base_url=args.base_url, api_key=args.api_key)
    transcriber = OfflineTranscriber(
        asr_client=asr_client,
        num_threads=args.num_threads,
        vad_target_segment_s=args.vad_target_segment_s,
        vad_max_segment_s=args.vad_max_segment_s,
        aligner_base_url=os.environ.get("QWEN3_ALIGNER_BASE_URL", ""),
        aligner_api_key=os.environ.get("QWEN3_ALIGNER_API_KEY", args.api_key or os.environ.get("OPENAI_API_KEY", "")),
        aligner_model=os.environ.get("QWEN3_ALIGNER_MODEL", "Qwen3-ForcedAligner-0.6B"),
        aligner_timeout_s=int(os.environ.get("QWEN3_ALIGNER_TIMEOUT_S", "120")),
        aligner_timestamp_segment_time_ms=int(os.environ.get("QWEN3_ALIGNER_TIMESTAMP_SEGMENT_TIME_MS", "80")),
    )

    result = transcriber.transcribe_file(
        input_file=input_file,
        context=args.context,
        use_forced_aligner=args.use_forced_aligner,
    )

    if not args.silence:
        print(f"Detected Language: {result['language']}")
        print(f"Segments: {result['segment_count']}")
        print(f"Full Transcription: {result['text']}")
        if result["forced_aligner"]["requested"]:
            print(
                "Forced aligner status: "
                f"available={result['forced_aligner']['available']}, "
                f"message={result['forced_aligner']['message']}"
            )

    if input_file.startswith(("http://", "https://")):
        base_name = os.path.splitext(os.path.basename(urlparse(input_file).path))[0] or "transcription"
        save_file = base_name + ".txt"
    else:
        save_file = os.path.splitext(input_file)[0] + ".txt"
    with open(save_file, "w", encoding="utf-8") as f:
        f.write((result.get("language") or "") + "\n")
        f.write((result.get("text") or "") + "\n")
    print(f'Transcription saved to "{save_file}"')

    if args.save_srt:
        subtitles = []
        for idx, segment in enumerate(result.get("segments", []), start=1):
            subtitles.append(
                srt.Subtitle(
                    index=idx,
                    start=timedelta(seconds=float(segment["start_sec"])),
                    end=timedelta(seconds=float(segment["end_sec"])),
                    content=segment["text"],
                )
            )
        srt_file = os.path.splitext(save_file)[0] + ".srt"
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write(srt.compose(subtitles))
        print(f'SRT subtitles saved to "{srt_file}"')


if __name__ == "__main__":
    main()
