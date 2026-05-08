import os
import io
import librosa
import subprocess
import numpy as np
import soundfile as sf

from silero_vad import get_speech_timestamps


WAV_SAMPLE_RATE = 16000


def _load_audio_with_pyav(file_path: str) -> np.ndarray:
    import av

    chunks = []
    with av.open(file_path) as container:
        stream = next((s for s in container.streams if s.type == 'audio'), None)
        if stream is None:
            raise RuntimeError('No audio stream found.')

        resampler = av.audio.resampler.AudioResampler(
            format='fltp',
            layout='mono',
            rate=WAV_SAMPLE_RATE,
        )
        for frame in container.decode(stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            if not isinstance(resampled, list):
                resampled = [resampled]
            for audio_frame in resampled:
                array = audio_frame.to_ndarray()
                chunks.append(np.asarray(array, dtype=np.float32).reshape(-1))

        flushed = resampler.resample(None)
        if flushed is not None:
            if not isinstance(flushed, list):
                flushed = [flushed]
            for audio_frame in flushed:
                array = audio_frame.to_ndarray()
                chunks.append(np.asarray(array, dtype=np.float32).reshape(-1))

    if not chunks:
        raise RuntimeError('Decoded audio is empty.')
    return np.concatenate(chunks).astype(np.float32, copy=False)


def load_audio(file_path: str) -> np.ndarray:
    try:
        if file_path.startswith(("http://", "https://")):
            raise ValueError("Using ffmpeg to load remote file.")
        # Try librosa first, because it is usually faster for standard formats.
        wav_data, _ = librosa.load(file_path, sr=WAV_SAMPLE_RATE, mono=True)
        return wav_data
    except Exception as e:
        print(e)
        try:
            return _load_audio_with_pyav(file_path)
        except Exception as pyav_e:
            # After librosa/PyAV fail, use a more powerful ffmpeg as a backup.
            try:
                command = [
                    'ffmpeg',
                    '-i', file_path,
                    '-ar', str(WAV_SAMPLE_RATE),
                    '-ac', '1',
                    '-c:a', 'pcm_s16le',
                    '-f', 'wav',
                    '-'
                ]
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                stdout_data, stderr_data = process.communicate()

                if process.returncode != 0:
                    raise RuntimeError(f"FFmpeg error processing local file: {stderr_data.decode('utf-8', errors='ignore')}")

                with io.BytesIO(stdout_data) as data_io:
                    wav_data, sr = sf.read(data_io, dtype='float32')

                return wav_data
            except Exception as ffmpeg_e:
                raise RuntimeError(
                    f"Failed to load audio from local file '{file_path}' with librosa, PyAV, or ffmpeg. "
                    f"PyAV error: {pyav_e}; ffmpeg error: {ffmpeg_e}"
                )


def process_vad(wav: np.ndarray, worker_vad_model, segment_threshold_s: int = 120, max_segment_threshold_s: int = 180) -> list[np.ndarray]:
    try:
        vad_params = {
            'sampling_rate': WAV_SAMPLE_RATE,
            'return_seconds': False,
            'min_speech_duration_ms': 500,
            'min_silence_duration_ms': 300
        }

        speech_timestamps = get_speech_timestamps(
            wav,
            worker_vad_model,
            **vad_params
        )

        if not speech_timestamps:
            raise ValueError("No speech segments detected by VAD.")

        # Build richer candidate split points:
        # - every speech segment start and end
        # - midpoints of gaps between consecutive speech segments
        potential_split_points = {0, len(wav)}
        for i, ts in enumerate(speech_timestamps):
            potential_split_points.add(ts['start'])
            potential_split_points.add(ts['end'])
            if i > 0:
                prev_end = speech_timestamps[i - 1]['end']
                curr_start = ts['start']
                if curr_start > prev_end:
                    potential_split_points.add((prev_end + curr_start) // 2)

        sorted_potential_splits = sorted(potential_split_points)

        final_split_points = {0, len(wav)}
        segment_threshold_samples = segment_threshold_s * WAV_SAMPLE_RATE
        target_time = segment_threshold_samples
        while target_time < len(wav):
            closest_point = min(sorted_potential_splits, key=lambda p: abs(p - target_time))
            final_split_points.add(closest_point)
            target_time += segment_threshold_samples
        final_ordered_splits = sorted(final_split_points)

        max_segment_threshold_samples = max_segment_threshold_s * WAV_SAMPLE_RATE
        new_split_points = [0]

        # Make sure that each audio segment does not exceed max_segment_threshold_s
        for i in range(1, len(final_ordered_splits)):
            start = final_ordered_splits[i - 1]
            end = final_ordered_splits[i]
            segment_length = end - start

            if segment_length <= max_segment_threshold_samples:
                new_split_points.append(end)
            else:
                num_subsegments = int(np.ceil(segment_length / max_segment_threshold_samples))
                subsegment_length = segment_length / num_subsegments

                for j in range(1, num_subsegments):
                    split_point = start + j * subsegment_length
                    new_split_points.append(split_point)

                new_split_points.append(end)

        segmented_wavs = []
        for i in range(len(new_split_points) - 1):
            start_sample = int(new_split_points[i])
            end_sample = int(new_split_points[i + 1])
            segmented_wavs.append((start_sample, end_sample, wav[start_sample:end_sample]))
        return segmented_wavs

    except Exception as e:
        segmented_wavs = []
        total_samples = len(wav)
        max_chunk_size_samples = max_segment_threshold_s * WAV_SAMPLE_RATE

        for start_sample in range(0, total_samples, max_chunk_size_samples):
            end_sample = min(start_sample + max_chunk_size_samples, total_samples)
            segment = wav[start_sample:end_sample]
            if len(segment) > 0:
                segmented_wavs.append((start_sample, end_sample, segment))

        return segmented_wavs


def save_audio_file(wav: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, wav, WAV_SAMPLE_RATE)
