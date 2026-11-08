#!/usr/bin/env python3
"""Report old human-document candidates without changing source documents."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

SKIP = {'.git', 'node_modules', 'target', 'dist', 'build', '.venv', 'venv', '__pycache__', '.cache', '.claude', '.codex', '.agents', 'skills', 'ai-config', 'vendor', 'site-packages', 'fixtures', 'testdata', 'logs'}
MAINTAINED = re.compile(r'^(README|AGENTS|CLAUDE|CONTRIBUTING|LICENSE|COPYING|CHANGELOG|TONE|PORTABILITY)([._-]|$)', re.I)
TASK = re.compile(r'(DOIT|TODO|DONE|PLAN|STATUS|review|research|retrospective|migration|report|audit|^\d{4}-\d{2})', re.I)
STATE = re.compile(r'^\s*(?:State|Status|Resolved|Completed):\s*(.{1,160})', re.I | re.M)


def git_state(path, cache, workspace):
    root = next((p for p in path.parents if p.is_relative_to(workspace) and (p / '.git').exists()), None)
    if root is None:
        return {'repository': None, 'disposition': 'outside-git'}
    if root not in cache:
        probe = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
        if probe.returncode:
            cache[root] = None
            return {'repository': str(root), 'disposition': 'invalid-git-metadata', 'git_error': probe.stderr.strip()}
        tracked = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
        status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain=v1', '-z', '--untracked-files=no']).decode().split('\0')
        changed = {}
        index = 0
        while index < len(status):
            row = status[index]
            if row:
                changed[row[3:]] = row[:2]
                if 'R' in row[:2] or 'C' in row[:2]:
                    index += 1
            index += 1
        cache[root] = set(tracked), changed
    if cache[root] is None:
        return {'repository': str(root), 'disposition': 'invalid-git-metadata'}
    tracked, changed = cache[root]
    relative = path.relative_to(root).as_posix()
    disposition = changed.get(relative, 'tracked-clean' if relative in tracked else 'untracked')
    if disposition == 'untracked':
        ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', '--', relative])
        if ignored.returncode not in (0, 1):
            ignored.check_returncode()
        if ignored.returncode == 0:
            disposition = 'ignored'
    return {'repository': str(root), 'disposition': disposition}


def survey(root, days):
    documents = {}
    texts = {}
    now = time.time()
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = [d for d in dirs if d not in SKIP and not (Path(directory) / d).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or path.suffix.lower() not in {'.md', '.txt'}:
                continue
            relative = path.relative_to(root)
            if 'notes' not in relative.parts and not TASK.search(name) and not MAINTAINED.match(name):
                continue
            # Private identity/credential documents are not cleanup candidates.
            if any(word in name.lower() for word in ('credential', 'account-mapping', 'secret', 'token')):
                continue
            content = path.read_bytes()
            texts[path] = content.decode('utf-8', errors='replace')
            if not MAINTAINED.match(name):
                documents[path] = (path.stat(), hashlib.sha256(content).hexdigest())
    hashes = defaultdict(list)
    for path, (_, digest) in documents.items():
        hashes[digest].append(str(path))
    candidates = []
    git_cache = {}
    references = defaultdict(set)
    names = {p.name for p in documents}
    mention = re.compile(r'[\w.+-]+\.(?:md|txt)\b')
    special_names = {name for name in names if mention.fullmatch(name) is None}
    for source, text in texts.items():
        mentioned = set(mention.findall(text)) & names
        mentioned.update(name for name in special_names if name in text)
        for name in mentioned:
            references[name].add(str(source))
    for path, (stat, digest) in documents.items():
        age = (now - stat.st_mtime) / 86400
        if age <= days:
            continue
        state = STATE.search(texts[path])
        refs = sorted(references[path.name] - {str(path)})
        candidates.append({'path': str(path), 'age_days': round(age, 1), 'bytes': stat.st_size, **git_state(path, git_cache, root), 'exact_duplicates': [p for p in hashes[digest] if p != str(path)], 'reference_count': len(refs), 'reference_documents': refs, 'reference_method': 'basename mention; may be ambiguous', 'document_state_hint': state[1] if state else 'unknown', 'state_source': 'document text only; no live task verification'})
    candidates.sort(key=lambda row: (-row['bytes'], row['path']))
    return {'generated_at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(), 'root': str(root), 'older_than_days': days, 'mode': 'read-only candidates; age does not authorize deletion', 'reference_scope': 'Markdown/text human notes, task artifacts and maintained documents; excludes code, dependencies, ai-config, skills and private credentials', 'duplicate_scope': 'candidate-eligible documents including younger copies; excludes maintained docs', 'documents_examined': len(documents), 'candidates': candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'dev')
    parser.add_argument('--days', type=float, default=14)
    parser.add_argument('--output', type=Path, help='JSON destination; defaults to ~/dev/logs/<PT date>/document-cleanup-survey.json')
    args = parser.parse_args()
    if args.days < 0:
        parser.error('--days must be nonnegative')
    root = args.root.expanduser().resolve(strict=True)
    output = args.output or Path.home() / 'dev/logs' / datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d') / 'document-cleanup-survey.json'
    output = output.expanduser().resolve()
    if output.suffix != '.json' or not output.is_relative_to((Path.home() / 'dev/logs').resolve()):
        parser.error('--output must be a JSON report under ~/dev/logs')
    report = survey(root, args.days)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(output)
    print(f"{len(report['candidates'])} candidates; report: {output}")


if __name__ == '__main__':
    main()
