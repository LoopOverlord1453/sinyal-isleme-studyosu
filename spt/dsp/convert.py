"""Ham kablo tipindeki örnekleri sayısal bileşenlere ve fiziksel birime çevirme."""
from __future__ import annotations

import numpy as np

from spt.protocol.format import ChannelInfo, SampleType


def decode_int24(raw: np.ndarray, signed: bool) -> np.ndarray:
    """3 baytlık paketli (V3) değerleri int32'ye çevirir."""
    b = np.ascontiguousarray(raw).view(np.uint8).reshape(-1, 3).astype(np.int32)
    value = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
    if signed:
        value = (value ^ 0x800000) - 0x800000
    return value


def components(column: np.ndarray, stype: SampleType) -> list[np.ndarray]:
    """Bir kanalın ham verisini sayısal bileşenlerine ayırır: gerçel tiplerde [değer], karmaşıkta [I, Q].

    Mümkün olduğunda kopya değil görünüm döndürür (ör. cint16 için alan görünümü).
    """
    if stype.is_int24:
        return [decode_int24(column, stype.signed)]
    if stype.is_complex:
        if column.dtype.names:
            return [column["re"], column["im"]]
        return [column.real, column.imag]
    return [column]


def to_physical(values: np.ndarray, channel: ChannelInfo) -> np.ndarray:
    return values.astype(np.float64) * channel.scale + channel.offset


def physical_column(column: np.ndarray, stype: SampleType, channel: ChannelInfo) -> np.ndarray:
    """Kanalın fiziksel değerleri: gerçel tiplerde float64, karmaşık tiplerde complex128."""
    parts = [to_physical(c, channel) for c in components(column, stype)]
    if len(parts) == 2:
        return parts[0] + 1j * parts[1]
    return parts[0]
