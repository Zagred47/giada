"""Byte compatibility and atomic failure semantics of large JSON export."""
import json,os,tempfile,unittest
from pathlib import Path
from research_memory.mirror import ROOT,write_json
class StreamingExportTests(unittest.TestCase):
    def test_identical_legacy_bytes(self):
        data={'unicode':'🔬 equilibrio τ e calcio','rows':[{'value':1.234567890123,'text':'x\ny','none':None,'bool':True}]*2000}
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'artifacts') as directory:
            path=Path(directory)/'snapshot.json';write_json(path,data)
            expected=(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n').replace('\n',os.linesep).encode('utf-8')
            self.assertEqual(path.read_bytes(),expected);self.assertEqual(json.loads(path.read_text(encoding='utf-8')),data)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])
    def test_failed_export_keeps_previous_file(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'artifacts') as directory:
            path=Path(directory)/'snapshot.json';write_json(path,{'valid':True});before=path.read_bytes()
            with self.assertRaises(ValueError):write_json(path,{'invalid':float('nan')})
            self.assertEqual(path.read_bytes(),before);self.assertEqual(list(Path(directory).glob('*.tmp')),[])
if __name__=='__main__':unittest.main()
