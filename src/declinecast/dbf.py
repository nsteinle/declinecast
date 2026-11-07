"""A minimal reader for the dBASE (.dbf) attribute table inside a zipped shapefile.

ECMC publishes its well master and directional surveys as shapefiles. Only the attribute
table is needed (spud date, depths, location, pad), so this reads the .dbf directly with the
standard library instead of depending on a GIS stack. Field values are returned as stripped
text; callers convert the columns they use.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import pandas as pd

__all__ = ["read_dbf"]


def read_dbf(zip_path: str | Path, member: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read ``member`` (a .dbf inside the zip at ``zip_path``) into a DataFrame of text columns.

    Deleted records (flag ``*``) are skipped. ``columns`` restricts the output to those fields,
    in that order; a missing field raises ``KeyError``."""
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        header = fh.read(32)
        n_records, header_len, record_len = struct.unpack("<IHH", header[4:12])
        descriptors = fh.read(header_len - 32)
        fields: list[tuple[str, int, int]] = []
        offset = 1  # byte 0 of each record is the deletion flag
        for i in range(0, len(descriptors), 32):
            block = descriptors[i:i + 32]
            if not block or block[0] == 0x0D:
                break
            name = block[:11].split(b"\0")[0].decode("latin-1")
            length = block[16]
            fields.append((name, offset, length))
            offset += length
        data = fh.read(n_records * record_len)
    known = {name for name, _, _ in fields}
    wanted = columns or [name for name, _, _ in fields]
    missing = [c for c in wanted if c not in known]
    if missing:
        raise KeyError(f"{member} has no fields {missing}")
    layout = [(name, off, length) for name, off, length in fields if name in set(wanted)]
    rows = []
    for r in range(n_records):
        rec = data[r * record_len:(r + 1) * record_len]
        if len(rec) < record_len or rec[:1] == b"*":
            continue
        rows.append({name: rec[off:off + length].decode("latin-1").strip()
                     for name, off, length in layout})
    return pd.DataFrame(rows, columns=wanted)
