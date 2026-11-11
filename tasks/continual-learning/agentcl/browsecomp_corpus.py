"""Build and query the offline BrowseComp+ index the judge searches.

BrowseComp-Plus ships queries and qrels; its corpus is published
separately and is not redistributable here, so ``fetch.py browsecomp``
takes a path to it and builds this index once. The index is SQLite FTS5
with a porter tokenizer -- deterministic, offline, and the same for
every task, which is what makes two runs comparable.

Ported from the AgentCL harness this conversion follows, minus its CLI.
"""

import csv
import json
import re
import sqlite3
from collections.abc import Iterable, Iterator
from pathlib import Path


def _document(row: dict, line_number: int) -> tuple[str, str, str]:
    doc_id = row.get("doc_id", row.get("id", row.get("_id")))
    title = row.get("title", "")
    text = row.get("text", row.get("contents", row.get("content", "")))
    if doc_id is None or not str(text).strip():
        raise ValueError(f"corpus row {line_number} lacks doc_id or text")
    return str(doc_id), str(title or ""), str(text)


def iter_documents(path: str | Path) -> Iterator[tuple[str, str, str]]:
    """Read JSONL or doc_id/title/text TSV without loading the corpus."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        if source.suffix.lower() in {".jsonl", ".json"}:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield _document(json.loads(line), line_number)
            return

        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, 1):
            if not row:
                continue
            if line_number == 1 and row[0].strip().lower() in {"doc_id", "id"}:
                continue
            if len(row) < 2:
                raise ValueError(
                    f"corpus TSV row {line_number} has fewer than 2 fields"
                )
            if len(row) == 2:
                yield str(row[0]), "", str(row[1])
            else:
                yield str(row[0]), str(row[1]), "\t".join(row[2:])


def build_index(
    documents: Iterable[tuple[str, str, str]],
    output: str | Path,
) -> int:
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite corpus index: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE documents (
                doc_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                title,
                text,
                content='documents',
                content_rowid='rowid',
                tokenize='porter unicode61'
            );
            """
        )
        count = 0
        batch = []
        for document in documents:
            batch.append(document)
            if len(batch) >= 10_000:
                connection.executemany(
                    "INSERT INTO documents(doc_id, title, text) VALUES (?, ?, ?)",
                    batch,
                )
                count += len(batch)
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO documents(doc_id, title, text) VALUES (?, ?, ?)",
                batch,
            )
            count += len(batch)
        if not count:
            raise ValueError("corpus is empty")
        connection.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return count
    except Exception:
        connection.close()
        target.unlink(missing_ok=True)
        raise
    finally:
        if connection:
            connection.close()


def qrel_doc_ids(paths: Iterable[str | Path]) -> set[str]:
    values = set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) < 3:
                    raise ValueError(f"invalid qrel row {path}:{line_number}")
                values.add(fields[2])
    return values


class BrowseCompCorpus:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"BrowseComp+ corpus index not found: {self.path}")
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        tables = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        if not {"documents", "documents_fts"} <= tables:
            raise ValueError("BrowseComp+ index lacks documents/documents_fts")

    def close(self) -> None:
        self.connection.close()

    def count(self) -> int:
        return int(
            self.connection.execute("SELECT count(*) FROM documents").fetchone()[0]
        )

    def missing_doc_ids(self, doc_ids: Iterable[str]) -> list[str]:
        missing = []
        for doc_id in sorted(set(doc_ids)):
            row = self.connection.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if row is None:
                missing.append(doc_id)
        return missing

    @staticmethod
    def _match_query(query: str) -> str:
        tokens = re.findall(r"[^\W_]+", query.casefold(), flags=re.UNICODE)
        tokens = [token.replace('"', '""') for token in tokens if len(token) > 1]
        return " OR ".join(f'"{token}"' for token in tokens[:64])

    def search(self, query: str, limit: int = 5) -> list[dict]:
        match = self._match_query(query)
        if not match:
            return []
        rows = self.connection.execute(
            """
            SELECT d.doc_id, d.title,
                   snippet(documents_fts, 1, '[', ']', ' ... ', 40) AS snippet,
                   bm25(documents_fts, 2.0, 1.0) AS score
            FROM documents_fts
            JOIN documents AS d ON d.rowid = documents_fts.rowid
            WHERE documents_fts MATCH ?
            ORDER BY score, d.doc_id
            LIMIT ?
            """,
            (match, max(1, min(int(limit), 20))),
        ).fetchall()
        return [
            {"doc_id": row[0], "title": row[1], "snippet": row[2], "score": row[3]}
            for row in rows
        ]

    def open(self, doc_id: str, max_chars: int = 12_000) -> dict | None:
        row = self.connection.execute(
            "SELECT title, text FROM documents WHERE doc_id = ?", (str(doc_id),)
        ).fetchone()
        if row is None:
            return None
        text = str(row[1])
        if len(text) > max_chars:
            text = text[: max_chars - 28] + "\n[document text truncated]"
        return {"doc_id": str(doc_id), "title": str(row[0]), "text": text}
