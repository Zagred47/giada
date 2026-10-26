"""Resume transport checks: no GPU and no live Kaggle writes."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('task41i_pipeline', ROOT/'scripts/task41i_kaggle_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class ResumeTransportTests(unittest.TestCase):
    def check_resume(self, already_current):
        with tempfile.TemporaryDirectory(dir=ROOT/'artifacts', prefix='task41i_transport_test_') as directory:
            root = Path(directory)
            (root/'artifacts').mkdir()
            destination = root/'external'
            destination.mkdir()
            (destination/'resume_dataset_created.json').write_text('{"created":true}')
            payload = b'original immutable checkpoint archive'
            archive = root/'stage.zip'
            archive.write_bytes(payload)
            receipt = {'local': str(archive), 'latest_checkpoint': 'checkpoint_512.pt',
                       'remote': {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}}

            class FakeApi:
                versions = 0
                remote = payload if already_current else b'previous phase'

                def dataset_download_file(self, dataset, name, path, **kwargs):
                    (Path(path)/name).write_bytes(self.remote)

                def dataset_create_version(self, folder, *args, **kwargs):
                    assert kwargs['delete_old_versions'] is False
                    self.versions += 1
                    self.remote = (Path(folder)/'resume.zip.bin').read_bytes()

                def dataset_status(self, *args):
                    raise AssertionError('Forbidden status endpoint must not be used')

            api = FakeApi()
            with patch.object(pipeline, 'ROOT', root), patch.object(pipeline, 'DEST', destination):
                pipeline.publish_resume(api, receipt)
            self.assertEqual(api.versions, 0 if already_current else 1)
            self.assertEqual((destination/'resume_dataset_readback/resume.zip.bin').read_bytes(), payload)

    def test_current_remote_checkpoint_is_not_republished(self):
        self.check_resume(True)

    def test_new_phase_is_versioned_and_verified_by_bytes(self):
        self.check_resume(False)


if __name__ == '__main__':
    unittest.main()
