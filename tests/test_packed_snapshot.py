import gzip
import pytest
from research_memory.packed_snapshot import pack_snapshot,restore_snapshot_if_missing


def test_lossless_and_no_overwrite(tmp_path):
    p=tmp_path/'snapshot.json';original=b'{"tables":{},"unicode":"\xc3\xa8"}\n';p.write_bytes(original)
    pack_snapshot(p);pack_snapshot(p);p.unlink()
    assert restore_snapshot_if_missing(p) and p.read_bytes()==original
    p.write_bytes(b'new local data')
    assert not restore_snapshot_if_missing(p) and p.read_bytes()==b'new local data'


def test_fail_closed(tmp_path):
    p=tmp_path/'snapshot.json';p.write_bytes(b'{}');pack_snapshot(p);p.unlink()
    p.with_suffix('.json.gz').write_bytes(gzip.compress(b'wrong'))
    with pytest.raises(ValueError):restore_snapshot_if_missing(p)
    assert not p.exists()
