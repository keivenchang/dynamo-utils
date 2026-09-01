from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import clean_opencode_db


def _database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT, time_updated INTEGER NOT NULL);
            CREATE TABLE event_sequence (aggregate_id TEXT PRIMARY KEY, seq INTEGER NOT NULL, owner_id TEXT);
            CREATE TABLE event (id TEXT PRIMARY KEY, aggregate_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE TABLE part (id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE TABLE session_context_epoch (session_id TEXT PRIMARY KEY, baseline TEXT NOT NULL, snapshot TEXT NOT NULL, baseline_seq INTEGER NOT NULL);
            CREATE TABLE session_input (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, prompt TEXT NOT NULL, delivery TEXT NOT NULL, admitted_seq INTEGER NOT NULL, promoted_seq INTEGER, time_created INTEGER NOT NULL);
            CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT NOT NULL, seq INTEGER NOT NULL, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL, data TEXT NOT NULL);
            CREATE TABLE session_share (session_id TEXT PRIMARY KEY, id TEXT NOT NULL, secret TEXT NOT NULL, url TEXT NOT NULL, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL);
            CREATE TABLE todo (session_id TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL, priority TEXT NOT NULL, position INTEGER NOT NULL, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL);
            """
        )
        old = 1
        recent = 9_999_999_999_999
        connection.executemany("INSERT INTO session VALUES (?, ?, ?)", [("old", None, old), ("recent", None, recent)])
        connection.execute("INSERT INTO event_sequence VALUES ('old', 1, NULL)")
        connection.execute("INSERT INTO event VALUES ('e-old', 'old', 1, 'message.updated.1', ?)", (json.dumps({"old": True}),))
        connection.execute("INSERT INTO message VALUES ('m-old', 'old', 1, 1, ?)", (json.dumps({"old": True}),))
        connection.execute("INSERT INTO part VALUES ('p-old', 'm-old', 'old', 1, 1, ?)", (json.dumps({"old": True}),))


def _add_child(connection: sqlite3.Connection, child_id: str, parent_id: str, updated: int) -> None:
    connection.execute("INSERT INTO session VALUES (?, ?)", (child_id, updated))
    connection.execute("UPDATE session SET time_updated = ? WHERE id = ?", (updated, parent_id))


def test_cleanup_dry_run_reports_old_payload_without_mutation(tmp_path: Path) -> None:
    database = tmp_path / "opencode.db"
    _database(database)
    cleaner = Path(__file__).parent / "clean_opencode_db.py"

    result = subprocess.run(
        [str(cleaner), "--db-path", str(database), "--keep-days", "1", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "candidates=1" in result.stdout
    assert "event_rows=1" in result.stdout
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 2


def test_cleanup_deletes_complete_old_session_and_vacuums(tmp_path: Path) -> None:
    database = tmp_path / "opencode.db"
    _database(database)
    cleaner = Path(__file__).parent / "clean_opencode_db.py"

    result = subprocess.run(
        [str(cleaner), "--db-path", str(database), "--keep-days", "1", "--vacuum"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT id FROM session").fetchall() == [("recent",)]
        assert connection.execute("SELECT COUNT(*) FROM event").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM message").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM part").fetchone()[0] == 0


def test_cleanup_refuses_active_database_unless_dry_run(tmp_path: Path, monkeypatch) -> None:
    database = tmp_path / "opencode.db"
    _database(database)
    monkeypatch.setattr(clean_opencode_db, "_active_opencode_pids", lambda _path: [1234])
    monkeypatch.setattr(sys, "argv", ["clean_opencode_db.py", "--db-path", str(database), "--keep-days", "1"])

    assert clean_opencode_db.main() == 3


def test_cleanup_keeps_old_parent_when_child_is_recent(tmp_path: Path) -> None:
    database = tmp_path / "opencode.db"
    _database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO session VALUES ('parent', NULL, 1)")
        connection.execute("INSERT INTO session VALUES ('child', 'parent', 9999999999999)")
    cleaner = Path(__file__).parent / "clean_opencode_db.py"

    result = subprocess.run(
        [str(cleaner), "--db-path", str(database), "--keep-days", "1", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "candidates=1" in result.stdout
