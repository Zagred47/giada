import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from research_memory.packed_database import pack, paths, restore_if_missing, digest


class PackedDatabaseTests(unittest.TestCase):
    def test_roundtrip_and_existing_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'memory.sqlite'
            with closing(sqlite3.connect(database)) as connection, connection:
                connection.execute('CREATE TABLE observations (value TEXT)')
                connection.execute('INSERT INTO observations VALUES (?)', ('preserved scientific data',))
            info = pack(database)
            archive, manifest = paths(database)
            self.assertEqual(digest(archive), info['compressed_sha256'])
            self.assertEqual(json.loads(manifest.read_text()), info)
            original = database.read_bytes()
            self.assertFalse(restore_if_missing(database))
            self.assertEqual(database.read_bytes(), original)
            database.unlink()
            self.assertTrue(restore_if_missing(database))
            self.assertEqual(digest(database), info['sqlite_sha256'])
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute('SELECT value FROM observations').fetchone()[0],
                                 'preserved scientific data')
            archive.write_bytes(b'corrupt')
            self.assertFalse(restore_if_missing(database))
            database.unlink()
            with self.assertRaises(ValueError):
                restore_if_missing(database)
            self.assertFalse(database.exists())

    def test_no_archive_does_not_create_empty_database(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'memory.sqlite'
            self.assertFalse(restore_if_missing(database))
            self.assertFalse(database.exists())
