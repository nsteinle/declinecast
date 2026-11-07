"""The data layer in DuckDB SQL; the simulator and the network stay in Python.

Each query is a ``.sql`` file next to this module. DuckDB runs in memory inside the Python
process, so there is no server and nothing to install beyond the ``duckdb`` package.

Two kinds of connection:

* ``connect(**tables)`` registers pandas DataFrames and switches external access off, so a
  query can only see the tables registered here. Everything after the panel build uses it,
  including ``declinecast sql``.
* ``connect_files()`` is used only to build the panel from the raw ECMC files. It may read local
  files, which DuckDB calls external access, but extension auto-loading and auto-installing are
  off, so it has no network access (no ``httpfs``).

The only text put into a query is file paths and column names, quoted; values are passed as
query parameters.

``tests/test_sql.py`` checks the panel build against an independent pandas implementation
(``tests/pandas_reference.py``) on a miniature copy of the raw files, and, when the real files
are present, on all 9.4 million rows.
"""

from __future__ import annotations

from importlib import resources

import duckdb
import pandas as pd

__all__ = ["connect", "connect_files", "load", "query", "ident", "literal"]

_NO_EXTENSIONS = {
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
}


def connect(**tables: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory connection with external access off and ``tables`` registered under their
    keyword names."""
    con = duckdb.connect(config={"enable_external_access": False, **_NO_EXTENSIONS})
    for name, frame in tables.items():
        con.register(name, frame)
    return con


def connect_files(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    """An in-memory connection that may read local files but loads no extensions."""
    config = dict(_NO_EXTENSIONS)
    if threads:
        config["threads"] = int(threads)
    return duckdb.connect(config=config)


def ident(name: str) -> str:
    """A column or table name quoted as a SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def literal(text: str) -> str:
    """A string quoted as a SQL literal (used for file paths, which cannot be parameters in a
    table function's file list)."""
    return "'" + str(text).replace("'", "''") + "'"


def load(name: str, **fields: str) -> str:
    """The text of ``<name>.sql`` with ``{placeholders}`` replaced by already-quoted text."""
    text = (resources.files(__package__) / f"{name}.sql").read_text()
    return text.format(**fields) if fields else text


def query(sql: str, **tables: pd.DataFrame) -> pd.DataFrame:
    """Run a read-only query over the given DataFrames (``declinecast sql``)."""
    with connect(**tables) as con:
        return con.execute(sql).df()
