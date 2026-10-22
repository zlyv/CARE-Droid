"""Check forensic selection, reference identities, literal redaction and non-causal output contracts."""
import unittest
from audit_common import canonical,stable_id,OUTPUT,PROJECT
from scan_code import reference,redact
from cohort import groups

class Contracts(unittest.TestCase):
    def test_canonical(self):self.assertEqual(canonical('Lfoo/A; -> run( I )V'),'lfoo/a;->run(i)v')
    def test_reference(self):self.assertEqual(reference('v0, v1, Ljava/lang/String;->substring(II)Ljava/lang/String;'),'ljava/lang/string;->substring(ii)ljava/lang/string;')
    def test_hash(self):self.assertEqual(stable_id('Lfoo/A;->run(I)V'),stable_id('lfoo/a;->run(i)v'))
    def test_literal_redaction(self):
        self.assertNotIn('secret-token',redact('const-string','v0, "secret-token"'))
        self.assertNotIn('secret-token',redact('const-string','v0, "android.permission.INTERNET secret-token"'))
    def test_output(self):self.assertEqual(OUTPUT.parent,PROJECT/'outputs')
    def test_frozen_cohort(self):self.assertEqual(sum(bool(r['category']) for r in groups().values()),73)

if __name__=='__main__':unittest.main()
