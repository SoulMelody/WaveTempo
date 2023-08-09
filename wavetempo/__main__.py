import itertools
import pathlib

import click
import librosa
import scipy as sp

from .wfd import WaveToneDataType, WaveToneFormatData, WaveTongTempoMaps


@click.command()
@click.argument('wfd_path', type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path))
@click.option('--dynamic', is_flag=True, default=False)
@click.option('--suffix', default='wav')
@click.option('--hop_length', default=512)
@click.option('--round_digits', default=1)
def get_tempos(wfd_path: pathlib.Path, dynamic: bool, suffix: str, hop_length: int, round_digits: int) -> None:
    audio_path = wfd_path.with_suffix(f'.{suffix}')
    if not audio_path.exists():
        raise click.BadParameter(f'File {audio_path} does not exist')
    wfd = WaveToneFormatData.parse_file(wfd_path)
    if wfd.start_offset < 0:
        raise click.BadParameter(f'Start offset must be positive, got {wfd.start_offset}')
    click.echo("loading audio file...")
    y, sr = librosa.load(audio_path, sr=None, offset=wfd.start_offset / 1000)
    click.echo("calculating tempo...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    prior = sp.stats.uniform(60, 240)
    tempo_data = []
    total_dur = wfd.start_offset / 1000
    ticks = 0
    if dynamic:
        tempo = librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr, aggregate=None, hop_length=hop_length,
            start_bpm=wfd.tempo, prior=prior, max_tempo=240
        )
        ticks_per_beat = 960
        for i, (bpm, group) in enumerate(itertools.groupby(tempo)):
            rounded_bpm = round(bpm, round_digits)
            if i:
                tempo_data.append({
                    "start": round(ticks),
                    "tempo": round(rounded_bpm * 10000),
                })
            duration = len(list(group)) * hop_length / sr
            total_dur += duration
            beats = duration / 60 * rounded_bpm
            ticks += beats * ticks_per_beat
    else:
        tempo = librosa.feature.tempo(
            onset_envelope=onset_env, sr=sr, hop_length=hop_length,
            start_bpm=wfd.tempo, prior=prior, max_tempo=240
        )
    tempo_data.insert(0, {
        "start": wfd.start_offset,
        "tempo": round(round(tempo[0], round_digits) * 10000),
    })
    tempo_data_index = None
    for i, (index, data) in enumerate(zip(wfd.indexes, wfd.data_bodies)):
        if int(index.data_type) == WaveToneDataType.TEMPO_MAP:
            tempo_data_index = i
            break
    if tempo_data_index is None:
        raise click.BadParameter(f'No tempo map found in {wfd_path}')
    tempo_map_content = WaveTongTempoMaps.build(tempo_data)
    wfd.data_bodies[i] = tempo_map_content
    wfd.indexes[i] = {
        "data_type": WaveToneDataType.TEMPO_MAP,
        "size": len(tempo_map_content)
    }
    wfd_path.write_bytes(WaveToneFormatData.build(wfd))


if __name__ == "__main__":
    get_tempos()