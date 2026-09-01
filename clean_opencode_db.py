#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove complete, inactive OpenCode sessions from its SQLite database."""

from __future__ import annotations

import argparse
import fcntl
import os
import sqlite3
import sys
import time
from pathlib import Path


DEFAULT_DB_PATH = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
REQUIRED_TABLES = {
    "event",
    "event_sequence",
    "message",
    "part",
    "session",
    "session_context_epoch",
    "session_input",
    "session_message",
    "session_share",
    "todo",
}
SESSION_CHILD_TABLES = (
    "session_context_epoch",
    "session_input",
    "session_message",
    "session_share",
    "todo",
)


def _active_opencode_pids(db_path: Path) -> list[int]:
    database_paths = {str(db_path), f"{db_path}-wal", f"{db_path}-shm"}
    active: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            executable = os.path.basename(os.readlink(entry / "exe")).casefold()
            command = (entry / "cmdline").read_bytes().split(b"\0", 1)[0].decode(errors="replace")
            if executable != "opencode" and os.path.basename(command).casefold() != "opencode":
                continue
            descriptors = entry / "fd"
            if any(os.path.realpath(link) in database_paths for link in descriptors.iterdir()):
                active.append(int(entry.name))
        except OSError:
            continue
    return sorted(active)


def _connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        uri = f"file:{path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
    else:
        connection = sqlite3.connect(path, timeout=5.0)
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    missing = REQUIRED_TABLES - tables
    if missing:
        raise RuntimeError(f"OpenCode database is missing required tables: {', '.join(sorted(missing))}")


def _old_session_ids(connection: sqlite3.Connection, cutoff_ms: int) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            """
            WITH RECURSIVE retained_ancestors(id) AS (
                SELECT parent_id FROM session
                WHERE time_updated >= ? AND parent_id IS NOT NULL
                UNION
                SELECT session.parent_id
                FROM session
                JOIN retained_ancestors ON session.id = retained_ancestors.id
                WHERE session.parent_id IS NOT NULL
            )
            SELECT id FROM session
            WHERE time_updated < ? AND id NOT IN (SELECT id FROM retained_ancestors)
            ORDER BY time_updated, id
            """,
            (cutoff_ms, cutoff_ms),
        )
    ]


def _in_clause(values: list[str]) -> tuple[str, list[str]]:
    return ",".join("?" for _ in values), values


def _payload_counts(connection: sqlite3.Connection, session_ids: list[str]) -> dict[str, tuple[int, int]]:
    if not session_ids:
        return {table: (0, 0) for table in ("event", "message", "part")}
    placeholders, parameters = _in_clause(session_ids)
    result: dict[str, tuple[int, int]] = {}
    for table, column in (("event", "aggregate_id"), ("message", "session_id"), ("part", "session_id")):
        count, payload = connection.execute(
            f"SELECT COUNT(*), COALESCE(SUM(length(data)), 0) FROM {table} "
            f"WHERE {column} IN ({placeholders})",
            parameters,
        ).fetchone()
        result[table] = (int(count), int(payload))
    return result


def _delete_sessions(connection: sqlite3.Connection, session_ids: list[str]) -> dict[str, int]:
    if not session_ids:
        return {}
    placeholders, parameters = _in_clause(session_ids)
    deleted: dict[str, int] = {}
    for table, column in (
        ("part", "session_id"),
        ("message", "session_id"),
        *[(table, "session_id") for table in SESSION_CHILD_TABLES],
        ("event", "aggregate_id"),
        ("event_sequence", "aggregate_id"),
        ("session", "id"),
    ):
        cursor = connection.execute(
            f"DELETE FROM {table} WHERE {column} IN ({placeholders})",
            parameters,
        )
        deleted[table] = cursor.rowcount
    return deleted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune old OpenCode sessions from its SQLite database.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--keep-days", type=int, required=True, help="Keep sessions updated within this many days")
    parser.add_argument("--dry-run", action="store_true", help="Report candidates without deleting rows")
    parser.add_argument("--vacuum", action="store_true", help="Compact the offline database after deletion")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.keep_days < 1:
        print("error: keep-days must be a positive integer", file=sys.stderr)
        return 2
    db_path = args.db_path.expanduser().resolve()
    if not db_path.is_file():
        print(f"opencode: database does not exist: {db_path}", file=sys.stderr)
        return 0
    active_pids = _active_opencode_pids(db_path)
    if active_pids and not args.dry_run:
        print(
            f"opencode: refusing mutation while active OpenCode processes hold the database: {','.join(map(str, active_pids))}",
            file=sys.stderr,
        )
        return 3
    cutoff_ms = int((time.time() - args.keep_days * 86400) * 1000)
    lock_path = db_path.with_name(f"{db_path.name}.cleanup.lock")
    try:
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            with _connect(db_path, read_only=args.dry_run) as connection:
                _validate_schema(connection)
                session_ids = _old_session_ids(connection, cutoff_ms)
                payloads = _payload_counts(connection, session_ids)
                payload_bytes = sum(payload for _, payload in payloads.values())
                print(
                    f"opencode: candidates={len(session_ids)} event_rows={payloads['event'][0]} "
                    f"message_rows={payloads['message'][0]} part_rows={payloads['part'][0]} "
                    f"payload_bytes={payload_bytes}"
                )
                if args.dry_run or not session_ids:
                    return 0
                active_pids = _active_opencode_pids(db_path)
                if active_pids:
                    print(
                        "opencode: refusing mutation while active OpenCode processes hold the database: "
                        + ",".join(map(str, active_pids)),
                        file=sys.stderr,
                    )
                    return 3
                connection.execute("BEGIN IMMEDIATE")
                active_pids = _active_opencode_pids(db_path)
                if active_pids:
                    connection.rollback()
                    print(
                        "opencode: refusing mutation because OpenCode opened the database during preparation: "
                        + ",".join(map(str, active_pids)),
                        file=sys.stderr,
                    )
                    return 3
                deleted = _delete_sessions(connection, session_ids)
                connection.commit()
                print(f"opencode: deleted={sum(deleted.values())} rows")
                if args.vacuum:
                    if _active_opencode_pids(db_path):
                        print("opencode: refusing offline vacuum because OpenCode reopened the database", file=sys.stderr)
                        return 3
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    connection.execute("VACUUM")
                    print("opencode: vacuum complete")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except (OSError, RuntimeError, sqlite3.Error) as error:
        print(f"opencode: cleanup failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
