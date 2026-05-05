import argparse
import math
import os
import sys
import numpy as np
import soundfile as sf


def _precompute_dft_table(size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Precompute sin/cos lookup table for DFT computation."""
    angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
    return np.cos(angles), np.sin(angles)


def read_audio_file(filepath: str) -> tuple[np.ndarray, int]:
    """
    Read an audio file and return audio data and sample rate.

    Supports WAV, FLAC, OGG, MP3 and other formats depending on
    available backends. Tries soundfile first, then scipy, then
    the standard library wave module.

    Parameters
    ----------
    filepath : str
        Path to the audio file.

    Returns
    -------
    audio_data : np.ndarray
        1D array of audio samples (float64, normalized to [-1, 1]).
    sample_rate : int
        Sample rate in Hz.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If no suitable audio backend is available or the file
        cannot be read.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    audio_data, sample_rate = sf.read(filepath, dtype='float64', always_2d=False)

    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]

    audio_data = audio_data.astype(np.float64)
    max_val = np.max(np.abs(audio_data))
    if max_val > 1.0:
        audio_data = audio_data / max_val

    return audio_data, sample_rate


def dfttempo_awd(
    onset_envelope: np.ndarray,
    sample_rate: float,
    hint_bpm: float = 0.0,
) -> tuple[float, float, np.ndarray]:
    """
    DFT-based tempo detection from onset strength envelope.

    It computes the DFT at candidate BPM frequencies and finds the BPM
    with maximum energy, combining fundamental and first harmonic.

    Parameters
    ----------
    onset_envelope : np.ndarray
        1D array of onset strength values over time.
    sample_rate : float
        Sample rate of the onset envelope (frames per second).
        This is typically: audio_sample_rate / hop_size.
    hint_bpm : float
        If > 0, refine around this BPM (±0.5 BPM, 0.01 steps, 101 candidates).
        If 0, search 60-239 BPM in 1-BPM steps (180 candidates).

    Returns
    -------
    bpm : float
        Detected beats per minute.
    beat_offset_ms : float
        Beat offset in milliseconds.
    energy_curve : np.ndarray
        Normalized energy curve for all BPM candidates.
    """
    n = len(onset_envelope)
    if n == 0:
        return 120.0, 0.0, np.array([])

    cos_table, sin_table = _precompute_dft_table(4096)
    table_size = 4096

    if hint_bpm > 0:
        bpm_candidates = hint_bpm + (np.arange(101) - 50) * 0.01
    else:
        bpm_candidates = 60.0 + np.arange(180, dtype=np.float64)
    num_candidates = len(bpm_candidates)
    energies = np.zeros(num_candidates)
    phases = np.zeros(num_candidates)

    for idx, bpm in enumerate(bpm_candidates):
        phase = 0.0
        sum_cos_fund = 0.0
        sum_sin_fund = 0.0
        sum_cos_harm = 0.0
        sum_sin_harm = 0.0

        step = table_size * 2.0 * bpm / (60.0 * sample_rate)

        for val in onset_envelope:
            idx_fund = (int(phase) >> 1) & 0xFFF
            idx_harm = int(phase) & 0xFFF

            sum_cos_fund += val * cos_table[idx_fund]
            sum_sin_fund += val * sin_table[idx_fund]
            sum_cos_harm += val * cos_table[idx_harm]
            sum_sin_harm += val * sin_table[idx_harm]

            phase += step

        fund_energy = math.sqrt(sum_cos_fund ** 2 + sum_sin_fund ** 2)
        harm_energy = math.sqrt(sum_cos_harm ** 2 + sum_sin_harm ** 2)
        total_energy = 0.5 * harm_energy + fund_energy

        energies[idx] = total_energy
        phases[idx] = math.atan2(-sum_sin_fund, sum_cos_fund)

    max_energy = np.max(energies)
    if max_energy == 0.0:
        max_energy = 1.0

    energy_curve = energies / max_energy
    best_idx = np.argmax(energies)
    best_bpm = bpm_candidates[best_idx]
    best_phase = phases[best_idx]

    if best_phase < 0:
        best_phase += 2.0 * math.pi

    offset_seconds = (
        -60.0 / best_bpm * best_phase / (2.0 * math.pi)
        + 1.0 / sample_rate
        + 0.03
    )

    beat_energies = np.zeros(4)
    beat_counts = np.zeros(4)

    for i, val in enumerate(onset_envelope):
        t = i / sample_rate
        beat_pos = int((t - offset_seconds) / (60.0 / best_bpm) + 0.5) % 4
        beat_energies[beat_pos] += val
        beat_counts[beat_pos] += 1

    for j in range(4):
        if beat_counts[j] > 0:
            beat_energies[j] /= beat_counts[j]

    e0, e1, e2, e3 = beat_energies
    scores = [
        e0 - 0.75 * e1 + 0.5 * e2 - 0.75 * e3,
        e1 - 0.75 * e2 + 0.5 * e3 - 0.75 * e0,
        e2 - 0.75 * e3 + 0.5 * e0 - 0.75 * e1,
        e3 - 0.75 * e0 + 0.5 * e1 - 0.75 * e2,
    ]
    best_beat = int(np.argmax(scores))

    beat_interval = 60.0 / best_bpm
    beat_offset_ms = (best_beat * beat_interval + offset_seconds) * 1000.0
    beat_offset_ms %= (beat_interval * 4.0 * 1000.0)

    return best_bpm, beat_offset_ms, energy_curve


def meas_key_tempo(
    beat_positions_ms: np.ndarray,
    audio_energy_curve: np.ndarray | None = None,
    time_sig_numerator: int = 4,
) -> tuple[int, int, int, np.ndarray]:
    """
    Measure key tempo from beat positions.

    Reimplementation of awlib.dll's MeasKeyTempo function.
    Given beat positions in milliseconds, finds the best constant BPM
    and beat offset.

    Parameters
    ----------
    beat_positions_ms : np.ndarray
        1D array of beat positions in milliseconds.
    audio_energy_curve : np.ndarray, optional
        Length-180 BPM energy curve from `dfttempo_awd` (indices map to
        BPM 60..239). Used jointly with the local beat-DFT energy via the
        original score `2 * audio_energy[i] + beat_dft_energy[i]` to pick
        the primary BPM, while the secondary BPM is the audio-energy max.
        If None, scoring falls back to the local beat-DFT energy alone.
    time_sig_numerator : int
        Time signature numerator (e.g., 4 for 4/4).

    Returns
    -------
    bpm : int
        Detected BPM (integer).
    beat_offset_ms : int
        Beat offset in milliseconds.
    secondary_bpm : int
        Secondary BPM candidate.
    energy_curve : np.ndarray
        Normalized local beat-DFT energy curve (length 180, BPM 60..239).
    """
    n_beats = len(beat_positions_ms)
    if n_beats < 2:
        return 120, 0, 120, np.array([])

    avg_interval_int = int(beat_positions_ms[-1] - beat_positions_ms[0]) // (n_beats - 1)
    if avg_interval_int <= 0:
        return 120, 0, 120, np.array([])

    approx_bpm = (avg_interval_int // 2 + 60000) // avg_interval_int

    bpm_min = approx_bpm - approx_bpm // 16
    if bpm_min < 60:
        bpm_min = 60
    width = approx_bpm // 8
    if bpm_min + width >= 240:
        width = 240 - bpm_min
    if width < 0:
        return 120, 0, 120, np.array([])
    bpm_max = bpm_min + width

    cos_table, sin_table = _precompute_dft_table(4096)
    table_size = 4096

    energies_full = np.zeros(180, dtype=np.float64)

    for bpm in range(bpm_min, bpm_max):
        step = table_size * 2.0 * bpm / (60.0 * 1000.0)
        sum_cos = 0.0
        sum_sin = 0.0
        for pos_ms in beat_positions_ms:
            phase = pos_ms * step
            table_idx = int(phase) & 0xFFF
            sum_cos += cos_table[table_idx]
            sum_sin += sin_table[table_idx]
        energies_full[bpm - 60] = sum_cos ** 2 + sum_sin ** 2

    max_energy = np.max(energies_full)
    if max_energy == 0.0:
        max_energy = 1.0
    energy_curve = energies_full / max_energy

    if audio_energy_curve is not None and len(audio_energy_curve) >= 180:
        a2 = np.asarray(audio_energy_curve[:180], dtype=np.float64)
    else:
        a2 = np.zeros(180, dtype=np.float64)

    best_bpm = 60
    best_score = 0.0
    sec_bpm = 60
    sec_max = 0.0
    for i in range(180):
        bpm_i = 60 + i
        score = 2.0 * a2[i] + energy_curve[i]
        if score > best_score:
            best_score = score
            best_bpm = bpm_i
        if a2[i] > sec_max:
            sec_max = a2[i]
            sec_bpm = bpm_i

    beat_offset_ms = meas_key_time(
        beat_positions_ms, best_bpm, time_sig_numerator
    )

    return best_bpm, beat_offset_ms, sec_bpm, energy_curve


def meas_key_time(
    beat_positions_ms: np.ndarray,
    bpm: float,
    time_sig_numerator: int = 4,
) -> int:
    """
    Compute beat offset given BPM and beat positions.

    Parameters
    ----------
    beat_positions_ms : np.ndarray
        Beat positions in milliseconds.
    bpm : float
        Beats per minute.
    time_sig_numerator : int
        Time signature numerator.

    Returns
    -------
    beat_offset_ms : int
        Beat offset in milliseconds.
    """
    n = len(beat_positions_ms)
    beat_interval_ms = 60000.0 / bpm

    error_sum = 0.0
    i = 0
    while i < n:
        if i + 3 < n:
            e0 = beat_positions_ms[i] - i * beat_interval_ms
            e1 = beat_positions_ms[i + 1] - (i + 1) * beat_interval_ms
            e2 = beat_positions_ms[i + 2] - (i + 2) * beat_interval_ms
            e3 = beat_positions_ms[i + 3] - (i + 3) * beat_interval_ms
            error_sum += e0 + e1 + e2 + e3
            i += 4
        else:
            error_sum += beat_positions_ms[i] - i * beat_interval_ms
            i += 1

    avg_error = error_sum / n
    measure_ms = time_sig_numerator * beat_interval_ms
    offset = int(avg_error - int(avg_error / measure_ms) * measure_ms)

    return offset


def detect_bpm_from_audio(
    audio_data: np.ndarray,
    audio_sample_rate: int,
    hop_size: int = 512,
    fft_size: int = 2048,
    hint_bpm: float = 0.0,
) -> tuple[float, float, np.ndarray]:
    """
    High-level BPM detection from audio data.

    This function implements the full pipeline:
    1. Compute STFT
    2. Compute spectral flux (onset strength envelope)
    3. Apply DFT-based tempo detection

    Parameters
    ----------
    audio_data : np.ndarray
        1D or 2D audio samples. If 2D, first channel is used.
    audio_sample_rate : int
        Audio sample rate in Hz.
    hop_size : int
        Hop size for STFT.
    fft_size : int
        FFT window size.
    hint_bpm : float
        Optional BPM hint to narrow search range.

    Returns
    -------
    bpm : float
        Detected BPM.
    beat_offset_ms : float
        Beat offset in milliseconds.
    energy_curve : np.ndarray
        Normalized BPM energy curve.
    """
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0] if audio_data.shape[1] <= audio_data.shape[0] else audio_data[0, :]
    audio_data = audio_data.astype(np.float64)

    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))

    n_frames = (len(audio_data) - fft_size) // hop_size + 1
    if n_frames < 1:
        return 120.0, 0.0, np.array([])

    window = np.hanning(fft_size)
    onset_env = np.zeros(n_frames, dtype=np.float64)

    prev_mag = None
    for i in range(n_frames):
        start = i * hop_size
        frame = audio_data[start:start + fft_size] * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)

        if prev_mag is not None:
            diff = mag - prev_mag
            diff = np.maximum(diff, 0.0)
            onset_env[i] = np.sum(diff)

        prev_mag = mag

    envelope_sample_rate = audio_sample_rate / hop_size

    return dfttempo_awd(onset_env, envelope_sample_rate, hint_bpm)


def detect_bpm_from_onset_envelope(
    onset_envelope: np.ndarray,
    envelope_sample_rate: float,
    hint_bpm: float = 0.0,
) -> tuple[float, float, np.ndarray]:
    """
    Detect BPM from a pre-computed onset strength envelope.

    Parameters
    ----------
    onset_envelope : np.ndarray
        Onset strength envelope.
    envelope_sample_rate : float
        Sample rate of the envelope (frames per second).
    hint_bpm : float
        Optional BPM hint.

    Returns
    -------
    bpm : float
    beat_offset_ms : float
    energy_curve : np.ndarray
    """
    return dfttempo_awd(onset_envelope, envelope_sample_rate, hint_bpm)


def find_beat_positions(
    onset_envelope: np.ndarray,
    envelope_sample_rate: float,
    bpm: float,
    beat_offset_ms: float = 0.0,
) -> np.ndarray:
    """
    Find beat positions in an onset envelope given a known BPM.

    Uses dynamic programming to find the best sequence of beat positions.

    Parameters
    ----------
    onset_envelope : np.ndarray
        Onset strength envelope.
    envelope_sample_rate : float
        Sample rate of the envelope.
    bpm : float
        Known BPM.
    beat_offset_ms : float
        Beat offset in milliseconds.

    Returns
    -------
    beat_times_ms : np.ndarray
        Beat positions in milliseconds.
    """
    n = len(onset_envelope)
    beat_interval_frames = envelope_sample_rate * 60.0 / bpm
    offset_frames = beat_offset_ms / 1000.0 * envelope_sample_rate

    max_beats = int(n / (beat_interval_frames * 0.5)) + 2
    beat_times_frames = offset_frames + np.arange(max_beats) * beat_interval_frames

    valid_beats = beat_times_frames[
        (beat_times_frames >= 0) & (beat_times_frames < n)
    ]

    if len(valid_beats) == 0:
        return np.array([])

    refined_beats = []
    search_radius = int(beat_interval_frames * 0.3)

    for center in valid_beats:
        lo = max(0, int(center) - search_radius)
        hi = min(n, int(center) + search_radius + 1)
        if lo < hi:
            best = lo + np.argmax(onset_envelope[lo:hi])
            refined_beats.append(best)

    return np.array(refined_beats) / envelope_sample_rate * 1000.0


def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry point for BPM detection.

    Usage:
        python bpm_detector.py <audio_file> [options]
        python bpm_detector.py <audio_file> --hint 140
        python bpm_detector.py <audio_file> --hop 1024 --fft 4096 --json
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bpm_detector.py song.wav
  python bpm_detector.py song.mp3 --hint 140
  python bpm_detector.py song.flac --hop 1024 --fft 4096
  python bpm_detector.py song.wav --json --beats
        """,
    )
    parser.add_argument(
        "audio_file",
        help="Path to the audio file (WAV, FLAC, MP3, OGG, etc.)",
    )
    parser.add_argument(
        "--hint", "-b",
        type=float,
        default=0.0,
        help="BPM hint to narrow search range (±50 BPM around hint)",
    )
    parser.add_argument(
        "--hop", "-p",
        type=int,
        default=512,
        help="Hop size for STFT (default: 512)",
    )
    parser.add_argument(
        "--fft", "-n",
        type=int,
        default=2048,
        help="FFT window size (default: 2048)",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--beats",
        action="store_true",
        help="Also output estimated beat positions",
    )
    parser.add_argument(
        "--energy", "-e",
        action="store_true",
        help="Also output the BPM energy curve",
    )

    args = parser.parse_args(argv)

    try:
        audio_data, audio_sr = read_audio_file(args.audio_file)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    duration = len(audio_data) / audio_sr
    if args.json:
        print(f"File: {args.audio_file}", file=sys.stderr)
        print(f"Sample rate: {audio_sr} Hz", file=sys.stderr)
        print(f"Duration: {duration:.2f}s", file=sys.stderr)
        print(f"Channels: 1 (mono)", file=sys.stderr)
    else:
        print(f"File: {args.audio_file}")
        print(f"Sample rate: {audio_sr} Hz")
        print(f"Duration: {duration:.2f}s")
        print(f"Channels: 1 (mono)")
        print(f"Hop size: {args.hop}, FFT size: {args.fft}")
        if args.hint > 0:
            print(f"BPM hint: {args.hint}")
        print()

    bpm, offset_ms, energy_curve = detect_bpm_from_audio(
        audio_data, audio_sr,
        hop_size=args.hop,
        fft_size=args.fft,
        hint_bpm=args.hint,
    )

    if args.json:
        import json
        result = {
            "bpm": round(bpm, 2),
            "beat_offset_ms": round(offset_ms, 2),
            "file": os.path.basename(args.audio_file),
            "sample_rate": audio_sr,
            "duration": round(duration, 2),
        }
        if args.energy:
            result["energy_curve"] = [round(v, 6) for v in energy_curve.tolist()]
        if args.beats:
            envelope_sample_rate = audio_sr / args.hop
            onset_env = _compute_onset_envelope(audio_data, args.hop, args.fft)
            beat_positions = find_beat_positions(
                onset_env, envelope_sample_rate, bpm, offset_ms
            )
            result["beat_positions_ms"] = [round(v, 2) for v in beat_positions.tolist()]
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Detected BPM: {bpm:.2f}")
        print(f"Beat offset: {offset_ms:.2f} ms")
        if args.energy:
            print(f"Energy curve (normalized):")
            for i, v in enumerate(energy_curve):
                print(f"  [{i:3d}] {v:.6f}")
        if args.beats:
            envelope_sample_rate = audio_sr / args.hop
            onset_env = _compute_onset_envelope(audio_data, args.hop, args.fft)
            beat_positions = find_beat_positions(
                onset_env, envelope_sample_rate, bpm, offset_ms
            )
            print(f"Beat positions ({len(beat_positions)} beats):")
            for i, bt in enumerate(beat_positions):
                print(f"  Beat {i:4d}: {bt:8.2f} ms")

    return 0


def _compute_onset_envelope(
    audio_data: np.ndarray,
    hop_size: int,
    fft_size: int,
) -> np.ndarray:
    """Compute onset strength envelope via spectral flux."""
    n_frames = (len(audio_data) - fft_size) // hop_size + 1
    if n_frames < 1:
        return np.array([])

    window = np.hanning(fft_size)
    onset_env = np.zeros(n_frames, dtype=np.float64)
    prev_mag = None

    for i in range(n_frames):
        start = i * hop_size
        frame = audio_data[start:start + fft_size] * window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec)
        if prev_mag is not None:
            diff = mag - prev_mag
            diff = np.maximum(diff, 0.0)
            onset_env[i] = np.sum(diff)
        prev_mag = mag

    return onset_env


if __name__ == "__main__":
    sys.exit(main())
