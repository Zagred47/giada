"""Lossless Git snapshot of the authoritative local SQLite database.

Existing local databases are never replaced. Packing requires a quiescent,
verified database; a read transaction and SQLite backup include all commits.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from contextlib import closing
import uuid


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def paths(database):
    database = Path(database)
    return database.with_suffix('.sqlite.gz'), database.with_suffix('.sqlite.manifest.json')


def restore_if_missing(database):
    database = Path(database)
    if database.exists():
        return False
    archive, manifest = paths(database)
    if not archive.exists() and not manifest.exists():
        return False
    info = json.loads(manifest.read_text(encoding='utf-8'))
    if info['format'] != 'sqlite-gzip-lossless-v1' or digest(archive) != info['compressed_sha256']:
        raise ValueError('Packed SQLite archive hash mismatch; no database restored')
    with tempfile.TemporaryDirectory(dir=database.parent) as directory:
        restored = Path(directory) / 'verified.sqlite'
        with gzip.open(archive, 'rb') as source, restored.open('wb') as target:
            shutil.copyfileobj(source, target)
        if restored.stat().st_size != info['sqlite_bytes'] or digest(restored) != info['sqlite_sha256']:
            raise ValueError('Packed SQLite content hash mismatch; no database restored')
        with closing(sqlite3.connect(restored)) as connection:
            if connection.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Packed SQLite integrity check failed')
        # Inherit the final parent ACL rather than the private staging ACL.
        publication = database.parent / ('restored-' + uuid.uuid4().hex + '.sqlite.tmp')
        try:
            with restored.open('rb') as source, publication.open('xb') as target:
                shutil.copyfileobj(source, target)
            # Exclusive link: atomic, never replaces a concurrently created DB.
            try:
                os.link(publication, database)
            except FileExistsError:
                return False
        finally:
            publication.unlink(missing_ok=True)
    return True


def pack(database):
    database = Path(database)
    archive, manifest = paths(database)
    with tempfile.TemporaryDirectory(dir=database.parent) as directory:
        backup = Path(directory) / 'snapshot.sqlite'
        with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as source:
            with closing(sqlite3.connect(backup)) as target:
                source.backup(target)
                if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('SQLite integrity check failed; snapshot not published')
                if target.execute('PRAGMA foreign_key_check').fetchone() is not None:
                    raise ValueError('SQLite foreign key check failed; snapshot not published')
        compressed = Path(directory) / 'snapshot.gz'
        with backup.open('rb') as source, compressed.open('wb') as target:
            with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as stream:
                shutil.copyfileobj(source, stream)
        info = dict(format='sqlite-gzip-lossless-v1', sqlite_sha256=digest(backup),
                    sqlite_bytes=backup.stat().st_size, compressed_sha256=digest(compressed),
                    compressed_bytes=compressed.stat().st_size)
        # Publish from siblings inheriting the data directory's ACL. Moving
        # files out of a private TemporaryDirectory otherwise carries its
        # restrictive Windows ACL, preventing the human user's Git from reading.
        publication = database.parent / ('packed-' + uuid.uuid4().hex + '.gz.tmp')
        temporary_manifest = database.parent / ('packed-' + uuid.uuid4().hex + '.json.tmp')
        try:
            with compressed.open('rb') as source, publication.open('xb') as target:
                shutil.copyfileobj(source, target)
            with temporary_manifest.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(info, indent=2) + '\n')
            # A reader in the small publication interval fails closed on hash.
            os.replace(publication, archive)
            os.replace(temporary_manifest, manifest)
        finally:
            publication.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
    return info


if __name__ == '__main__':
    from .mirror import DEFAULT_DB, Mirror
    mirror = Mirror()
    if not mirror.verify()['valid']:
        raise ValueError('Mirror verification failed; snapshot not published')
    print(json.dumps(pack(DEFAULT_DB)))
