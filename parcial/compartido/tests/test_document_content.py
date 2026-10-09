"""Reject real content omissions while tolerating glyph and reading-order differences."""

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_document_content.py"
SPEC = importlib.util.spec_from_file_location("audit_document_content", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class ContentAuditTests(unittest.TestCase):
    def test_ligatures_controls_and_table_order_do_not_create_false_token_loss(self):
        source = "justi\x1ccaciones\nBaja 340 0.1582\nMedia 69 0.0582\n"
        target = "Media 69 0.0582\njustificaciones\nBaja 340 0.1582\n"
        self.assertEqual(AUDIT.comparison(source, target)["missing_token_instances"], 0)

    def test_missing_repeated_metric_is_detected_even_when_it_remains_in_prose(self):
        source = "Texto 0.2107.\nCLAHE 0.2107 0.9972 0.3412 0.7102\n"
        target = "Texto 0.2107.\n"
        missing = AUDIT.comparison(source, target)["missing_tokens"]
        self.assertEqual(missing["0.2107"], 1)
        self.assertEqual(missing["0.9972"], 1)

    def test_xml_preserves_explicit_spaces_and_line_breaks_between_spans(self):
        xml = b'''<office:document-content
          xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
          xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
          <office:body><text:p>398<text:s/><text:span>notas</text:span>
          <text:line-break/>sin inventar</text:p></office:body></office:document-content>'''
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.odt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("content.xml", xml)
            text, _ = AUDIT.odt_text(path)
        self.assertIn("398 notas", text)
        self.assertIn("\nsin inventar", text)

    def test_removal_of_control_glyph_line_is_a_real_xml_omission(self):
        source = "aceptaron como opcionales: 398 permanecen ausentes; no se inventan justi\x1ccaciones"
        target = "otras notas"
        result = AUDIT.comparison(source, target)
        self.assertIn("398", result["missing_tokens"])
        self.assertIn("justificaciones", result["missing_tokens"])


if __name__ == "__main__":
    unittest.main()
