"""Exercise destructive-operation guards using isolated temporary fixtures."""

import tempfile
import unittest
from pathlib import Path

from migrate_structure import apply, contained, digest, validate


class MigrationSafetyTests(unittest.TestCase):
    """Require byte identity, bounded targets and a separate D2 authorization."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "parcial/source.txt"
        self.source.parent.mkdir()
        self.source.write_bytes(b"frozen\r\nbytes\r\n")
        self.row = {"source": "parcial/source.txt", "destination": "parcial/compartido/destination.txt",
                    "sha256": digest(self.source), "action": "move", "frozen": False}
        self.plan = {"stage": "D1", "worktree": str(self.root), "operations": [self.row]}

    def test_dry_run_preserves_bytes_and_creates_no_destination(self) -> None:
        before = self.source.read_bytes()
        self.assertEqual(validate(self.plan, self.root)["mutations"], 0)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertFalse((self.root / self.row["destination"]).exists())

    def test_rejects_traversal_and_absolute_targets(self) -> None:
        for name in ("parcial/../../outside.txt", str(self.source), "outside.txt"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                contained(self.root, name)

    def test_rejects_destination_collision_without_overwrite(self) -> None:
        self.row["destination"] = self.row["source"]
        with self.assertRaisesRegex(ValueError, "already exists"):
            validate(self.plan, self.root)
        self.assertEqual(self.source.read_bytes(), b"frozen\r\nbytes\r\n")

    def test_rejects_stale_source_hash(self) -> None:
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            validate(self.plan, self.root)

    def test_rejects_wrong_worktree(self) -> None:
        self.plan["worktree"] = str(self.root / "another")
        with self.assertRaisesRegex(ValueError, "another worktree"):
            validate(self.plan, self.root)

    def test_requires_original_bytes_before_retiring_a_copy(self) -> None:
        mirror = self.root / "mirror.jpg"
        mirror.write_bytes(self.source.read_bytes())
        self.row.update(action="retire_copy", destination="", reconstruction={"method": "copy_bytes",
                        "source": str(mirror), "sha256": digest(mirror)})
        self.assertEqual(validate(self.plan, self.root)["operations"]["retire_copy"], 1)
        mirror.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "Reconstruction changed"):
            validate(self.plan, self.root)
        self.assertTrue(self.source.exists())

    def test_d1_apply_is_refused_before_writing_recovery(self) -> None:
        with self.assertRaisesRegex(ValueError, "D1 cannot be applied"):
            apply(self.plan, self.root)
        self.assertTrue(self.source.exists())
        self.assertFalse((self.root / ".migration-recovery-d2").exists())


if __name__ == "__main__":
    unittest.main()
