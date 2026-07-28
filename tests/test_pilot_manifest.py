import unittest
from dataclasses import replace

from probekv.experiment_jobs import generate_e1_jobs
from probekv.manifest import synthetic_manifest
from probekv.pilot_manifest import pilot_manifest_audit, select_h1_pilot


class PilotManifestTests(unittest.TestCase):
    def test_selects_exact_train_only_stratified_pilot(self):
        base = synthetic_manifest(180, 20260726)
        datasets = ("MuSiQue", "2WikiMultiHopQA", "HotPotQA")
        cases = []
        for index, case in enumerate(base):
            dataset = datasets[index // 60]
            construction = (
                "corpus_repeat_pseudotime"
                if index % 60 < 30
                else "controlled_document_order"
            )
            cases.append(
                replace(
                    case,
                    case_id="%s:%s" % (dataset, case.case_id),
                    dataset=dataset,
                    document_id="%s:%s" % (dataset, case.document_id),
                    group_id="%s:%s" % (dataset, case.group_id),
                    split="train",
                    construction=construction,
                )
            )
        pilot = select_h1_pilot(cases)
        audit = pilot_manifest_audit(pilot)
        self.assertEqual(len(pilot), 150)
        self.assertTrue(audit["all_split_pilot"])
        for dataset in datasets:
            self.assertEqual(audit["datasets"][dataset]["cases"], 50)
            self.assertEqual(audit["datasets"][dataset]["natural"], 25)
            self.assertEqual(audit["datasets"][dataset]["controlled"], 25)
        jobs = generate_e1_jobs(
            pilot,
            total_layers=32,
            repair_ratios=(0.0, 0.05, 0.10, 0.16, 0.20, 0.30, 0.50, 0.75, 1.0),
            include_splits=("pilot",),
            anchor_fraction=0.20,
        )
        self.assertEqual(len(jobs), 9720)
        self.assertEqual(sum(job.reuse_layer == 5 for job in jobs), 5400)
        self.assertEqual(sum(job.reuse_layer != 5 for job in jobs), 4320)

    def test_non_train_cases_are_never_selected(self):
        cases = [
            replace(
                case,
                split="test",
                construction="controlled_document_order",
            )
            for case in synthetic_manifest(20, 20260726)
        ]
        with self.assertRaisesRegex(ValueError, "no train"):
            select_h1_pilot(cases)


if __name__ == "__main__":
    unittest.main()
