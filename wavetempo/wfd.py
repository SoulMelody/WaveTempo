from enum import IntEnum

from construct import (
    Byte,
    Bytes,
    BytesInteger,
    Const,
    GreedyRange,
    Padding,
    Struct,
    Terminated,
    this,
)
from construct import Enum as CSEnum

Int32sl = BytesInteger(4, swapped=True, signed=True)


class WaveToneDataType(IntEnum):
    VERSION_INFO = 0
    TEMPO_ANALYSIS_RESULTS = 2
    EXTENDED_INFO = 4
    LABEL_LIST = 6
    STEREO_SPECTRUM = 7
    L_MINUS_R_SPECTRUM = 8
    L_PLUS_R_SPECTRUM = 9
    LEFT_SPECTRUM = 10
    RIGHT_SPECTRUM = 11
    TEMPO_MAP = 12
    CHORD_DETECTION_RESULTS = 14
    RHYTHM_KEYMAP = 15
    NOTE_LIST = 16
    VOLUME_FOR_TEMP_ANALYSIS = 17
    FREQUENCY_ANALYSIS_RESULTS = 18
    TRACK_SETTINGS = 19


# Reference: http://ackiesound.ifdef.jp/doc/wfdformat/main.html


WaveToneTempoMap = Struct(
    "start" / Int32sl,
    "tempo" / Int32sl,
)
WaveToneTempoMaps = GreedyRange(WaveToneTempoMap)


WaveToneFormatData = Struct(
    "magic" / Const(b"WF"),
    "version" / Byte[2],
    Padding(4),
    Padding(4),
    "blocks_per_semitone" / Int32sl,
    "lowest_semitone" / Int32sl,
    "semitone_range" / Int32sl,
    "blocks_per_second" / Int32sl,
    Padding(4),
    "blocks_count" / Int32sl,
    "graph_data_bits" / Int32sl,
    "beats_display_flags" / Int32sl,
    "tempo" / Int32sl,
    "start_offset" / Int32sl,
    "time_signature" / Int32sl,
    "head_len" / Int32sl,
    "indexes" / Struct(
        "data_type" / CSEnum(
            Int32sl,
            WaveToneDataType
        ),
        "size" / Int32sl,
    )[this._root.head_len],
    "data_bodies" / Bytes(lambda this: this._root.indexes[this._index].size)[this._root.head_len],
    Terminated,
)