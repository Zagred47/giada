"""Record an uploaded backup after externally verified Drive metadata."""
import argparse
import json
from pathlib import Path
from .mirror import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    args = parser.parse_args()
    local = json.loads((ROOT/'drive_exports/latest_local.json').read_text())
    remote = json.loads(Path(args.metadata).read_text())
    folder = Path(local['folder'])
    for name, item in remote['files'].items():
        assert (folder/name).stat().st_size == int(item['size'])
        assert item['title'] == name and remote['folder_id'] in item['parent_ids']
    transport = local['transport']
    archive = remote['files'][transport['filename']]
    receipt = dict(format='sqlite-drive-backup-v1', snapshot_utc=local['snapshot_utc'],
        sqlite_sha256=local['sha256'], sqlite_bytes=local['bytes'],
        archive_sha256=transport['sha256'], archive_bytes=transport['bytes'],
        archive_format=transport['format'], sqlite_member=local['filename'],
        semantic_records=local['semantic_records'], drive_url=archive['url'],
        drive_folder_url='https://drive.google.com/drive/folders/'+remote['folder_id'],
        scope=remote['scope'], valid=True, metadata_names_sizes_parents_verified=True,
        single_original_sqlite_member_sha256_verified=transport['member_sha256_verified'],
        remote_archive_download_sha256_verified=remote.get('download_sha256') == transport['sha256'],
        files=remote['files'], integrity_check=local['integrity_check'],
        foreign_key_errors=local['foreign_key_violations'], airtable_accessed=False,
        sharing_changed=False, single_file=True)
    write_json(ROOT/'data/drive_backup_receipt.json', receipt)
    write_json(ROOT/'data/research_memory.sqlite.manifest.json', receipt)
    print(json.dumps(dict(valid=True, records=local['semantic_records'],
        remote_sha256_verified=receipt['remote_archive_download_sha256_verified'])))


if __name__ == '__main__':
    main()
