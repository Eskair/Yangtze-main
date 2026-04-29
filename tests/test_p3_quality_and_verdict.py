import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "src" / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import post_processing  # noqa: E402
import ai_expert_opinion  # noqa: E402


class TestP3QualityAndVerdict(unittest.TestCase):
    def test_quality_gate_fails_on_low_coverage(self):
        qg = post_processing.derive_quality_gates(
            total_q_count=10,
            selected_count=6,
            degraded_count=0,
            consistency_ratio=0.2,
        )
        self.assertFalse(qg["pass"])
        self.assertIn("selected_coverage_low", qg["reasons"])

    def test_quality_gate_fails_on_parse_success(self):
        qg = post_processing.derive_quality_gates(
            total_q_count=10,
            selected_count=10,
            degraded_count=3,
            consistency_ratio=0.2,
        )
        self.assertFalse(qg["pass"])
        self.assertIn("parse_success_low", qg["reasons"])

    def test_verdict_insufficient_evidence_when_gate_fails(self):
        dims = {
            "team": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "objectives": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "strategy": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "innovation": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "feasibility": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
        }
        overall = ai_expert_opinion.build_overall_from_dims(
            dim_blocks=dims,
            metrics_overall={
                "overall_score": 0.75,
                "overall_confidence": 0.8,
                "quality_gates": {"pass": False, "reasons": ["parse_success_low"]},
            },
            metrics_dims={
                "innovation": {"avg": 0.7},
                "feasibility": {"avg": 0.7},
            },
        )
        self.assertEqual(overall["verdict"], "INSUFFICIENT_EVIDENCE")

    def test_verdict_go_when_gate_passes_and_scores_high(self):
        dims = {
            "team": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "objectives": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "strategy": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "innovation": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
            "feasibility": {"summary": "x", "strengths": [], "concerns": [], "recommendations": []},
        }
        overall = ai_expert_opinion.build_overall_from_dims(
            dim_blocks=dims,
            metrics_overall={
                "overall_score": 0.75,
                "overall_confidence": 0.8,
                "quality_gates": {"pass": True, "reasons": []},
            },
            metrics_dims={
                "innovation": {"avg": 0.7},
                "feasibility": {"avg": 0.7},
            },
        )
        self.assertEqual(overall["verdict"], "GO")


if __name__ == "__main__":
    unittest.main()
