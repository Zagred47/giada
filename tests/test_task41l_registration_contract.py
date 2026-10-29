"""Schema-only regression guard; no writes to the research database."""
import unittest
from research_memory.mirror import Contract
from research_memory.register_task41l_result import EVIDENCE_POSITIVE
from research_memory.register_task41m import BASELINE_ROLE

class EvidenceVocabularyTest(unittest.TestCase):
    def test_positive_outcome_uses_schema_vocabulary(self):
        contract=Contract()
        contract.normalize('evidence',{'Esito':EVIDENCE_POSITIVE},partial=True)
        self.assertEqual(EVIDENCE_POSITIVE,'Sostiene')
        with self.assertRaises(ValueError):
            contract.normalize('evidence',{'Esito':'Supporta'},partial=True)

    def test_baseline_and_other41m_enum_values(self):
        contract=Contract()
        values={'arms':{'Ruolo':BASELINE_ROLE},'protocols':{'Modalità':'Ricomposizione'},
                'experiments':{'Stato':'Preregistrato'},'evaluations':{'Ruolo':'Primaria'},
                'runs':{'Stato':'Pianificata'},'decisions':{'Esito':'Continuare'}}
        for table,fields in values.items():contract.normalize(table,fields,partial=True)

if __name__=='__main__':unittest.main()
