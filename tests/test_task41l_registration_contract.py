"""Schema-only regression guard; no writes to the research database."""
import unittest
from research_memory.mirror import Contract
from research_memory.register_task41l_result import EVIDENCE_POSITIVE

class EvidenceVocabularyTest(unittest.TestCase):
    def test_positive_outcome_uses_schema_vocabulary(self):
        contract=Contract()
        contract.normalize('evidence',{'Esito':EVIDENCE_POSITIVE},partial=True)
        self.assertEqual(EVIDENCE_POSITIVE,'Sostiene')
        with self.assertRaises(ValueError):
            contract.normalize('evidence',{'Esito':'Supporta'},partial=True)

if __name__=='__main__':unittest.main()
