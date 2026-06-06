"""Storage helpers implementing the library's multi-layer architecture.

The plan calls for several physical layers with different access patterns:

  * cold array layer   -> Zarr chunked arrays (sketches, bitsets, centroids)
  * columnar table layer -> Parquet (catalogs, stats, edges, summaries)
  * query layer        -> DuckDB views over the Parquet files
  * vector / graph / bitmap index layers -> built in stage 13

This module centralizes the two cross-cutting concerns those layers share:
schema-checked writes and *atomic commits*. Every stage writes into a temporary
path and renames into place only after the bytes are durable, so an interrupted
run never leaves a half-written artifact that a downstream pipeline might read.

Heavy third-party imports (pyarrow, zarr, duckdb) are kept module-local-friendly
by importing at module top only what is light; zarr/duckdb are imported lazily
inside the functions that use them so that pure-metadata tooling can import this
module without those packages installed.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


@contextmanager
def atomic_path(final_path: Path, is_dir: bool = False) -> Iterator[Path]:
    """Yield a temporary path that is atomically renamed to ``final_path``.

    For files we write to ``name.tmp.<pid>`` in the same directory and
    ``os.replace`` on success. For directory artifacts (zarr stores) we build in
    a sibling temp directory and swap it in, removing any prior version.
    """
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if is_dir:
        tmp = Path(tempfile.mkdtemp(prefix=final_path.name + ".tmp.", dir=final_path.parent))
        try:
            yield tmp
            if final_path.exists():
                _rmtree(final_path)
            os.replace(tmp, final_path)
        finally:
            if tmp.exists():
                _rmtree(tmp)
    else:
        fd, tmp_name = tempfile.mkstemp(prefix=final_path.name + ".tmp.", dir=final_path.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            yield tmp
            os.replace(tmp, final_path)
        finally:
            if tmp.exists():
                tmp.unlink()


def _rmtree(path: Path) -> None:
    import shutil
    shutil.rmtree(path)


def write_parquet(
    rows: list[dict[str, Any]] | pa.Table,
    final_path: Path,
    schema: pa.Schema | None = None,
    partition_cols: list[str] | None = None,
) -> int:
    """Write ``rows`` to Parquet atomically. Returns the row count.

    If ``partition_cols`` is given the artifact is a *directory* of partitioned
    Parquet files (Hive layout), which lets downstream batch jobs scan only the
    partitions they need, per the large-scale performance rules.
    """
    if isinstance(rows, pa.Table):
        table = rows if schema is None else rows.cast(schema)
    else:
        table = _rows_to_table(rows, schema)

    if partition_cols:
        import pyarrow.dataset as ds
        with atomic_path(final_path, is_dir=True) as tmp:
            ds.write_dataset(
                table, str(tmp), format="parquet",
                partitioning=partition_cols, partitioning_flavor="hive",
            )
    else:
        with atomic_path(final_path, is_dir=False) as tmp:
            pq.write_table(table, str(tmp), compression="zstd")
    return table.num_rows


def _rows_to_table(rows: list[dict[str, Any]], schema: pa.Schema | None) -> pa.Table:
    if schema is None:
        return pa.Table.from_pylist(rows)
    if not rows:
        #Empty table with the declared schema, so consumers still see columns.
        arrays = [pa.array([], type=f.type) for f in schema]
        return pa.Table.from_arrays(arrays, schema=schema)
    columns = {f.name: [r.get(f.name) for r in rows] for f in schema}
    arrays = [pa.array(columns[f.name], type=f.type) for f in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def read_parquet(path: Path) -> pa.Table:
    """Read a Parquet file or partitioned dataset directory into a table."""
    path = Path(path)
    if path.is_dir():
        import pyarrow.dataset as ds
        return ds.dataset(str(path), format="parquet").to_table()
    return pq.read_table(str(path))


def validate_against_schema(table: pa.Table, schema: pa.Schema) -> list[str]:
    """Return a list of human-readable schema mismatches (empty == valid)."""
    problems: list[str] = []
    declared = {f.name: f.type for f in schema}
    present = {f.name: f.type for f in table.schema}
    for name, typ in declared.items():
        if name not in present:
            problems.append(f"missing column: {name}")
        elif not present[name].equals(typ):
            problems.append(f"type mismatch for {name}: {present[name]} != {typ}")
    return problems


#--------------------------------------------------------------------------
#Cold array layer (Zarr).
#--------------------------------------------------------------------------

def write_zarr_array(name_to_array: dict[str, "Any"], final_path: Path,
                     chunks: dict[str, Any] | None = None) -> None:
    """Write one or more named arrays into a single Zarr group, atomically.

    Unicode string arrays (e.g. unit-ID columns) are stored as fixed-width bytes
    so the store is portable across Zarr versions; ``read_zarr_str`` decodes them
    back to Python strings on read.
    """
    import zarr
    import numpy as np

    chunks = chunks or {}
    with atomic_path(final_path, is_dir=True) as tmp:
        root = zarr.open_group(str(tmp), mode="w")
        for name, arr in name_to_array.items():
            arr = np.asarray(arr)
            if arr.dtype.kind == "U":
                arr = np.char.encode(arr, "utf-8")  # -> 'S' bytes
            ds = root.create_array(name, shape=arr.shape, dtype=arr.dtype)
            ds[...] = arr


def read_zarr_group(path: Path):
    import zarr
    return zarr.open_group(str(path), mode="r")


def read_zarr_str(group, name: str) -> list[str]:
    """Read a string array from a Zarr group, decoding bytes back to str."""
    import numpy as np
    arr = np.asarray(group[name][:])
    if arr.dtype.kind == "S":
        return [x.decode("utf-8") for x in arr.tolist()]
    return [str(x) for x in arr.tolist()]


#--------------------------------------------------------------------------
#Query layer (DuckDB).
#--------------------------------------------------------------------------

def duckdb_connect(db_path: Path):
    import duckdb
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))
