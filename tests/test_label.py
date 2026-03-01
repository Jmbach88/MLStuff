import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from label import label_outcome, label_claim_types


class TestOutcomeLabeling:
    def test_plaintiff_win_judgment(self):
        text = "The court grants judgment for plaintiff. Damages are awarded in the amount of $1,000."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"
        assert result["confidence"] > 0.5

    def test_defendant_win_dismissed(self):
        text = "Plaintiff's complaint is dismissed with prejudice. Judgment for defendant."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"
        assert result["confidence"] > 0.5

    def test_mixed_partial(self):
        text = "Defendant's motion for summary judgment is granted in part and denied in part."
        result = label_outcome(text)
        assert result["label"] == "mixed"

    def test_unlabeled_no_signals(self):
        text = "This is a procedural order regarding scheduling of the pre-trial conference."
        result = label_outcome(text)
        assert result["label"] == "unlabeled"

    def test_plaintiff_win_summary_judgment(self):
        text = "Plaintiff's motion for summary judgment is GRANTED. The defendant is liable under the FDCPA."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_defendant_win_summary_judgment(self):
        text = "The court hereby grants defendant's motion for summary judgment. The case is dismissed."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"


class TestClaimTypeLabeling:
    def test_fdcpa_section_extraction(self):
        text = "The defendant violated \u00a71692e by making false representations about the debt."
        result = label_claim_types(text)
        assert "1692e" in result

    def test_fdcpa_multiple_sections(self):
        text = "Claims under \u00a71692e and \u00a71692f of the FDCPA were both asserted."
        result = label_claim_types(text)
        assert "1692e" in result
        assert "1692f" in result

    def test_fdcpa_section_with_subsection(self):
        text = "Section 1692e(5) prohibits threats to take action that cannot legally be taken."
        result = label_claim_types(text)
        assert "1692e" in result

    def test_tcpa_section(self):
        text = "The plaintiff alleged violations of 47 U.S.C. \u00a7227(b)(1)(A)."
        result = label_claim_types(text)
        assert "227" in result

    def test_fcra_section(self):
        text = "Defendant failed to conduct a reasonable investigation under 15 U.S.C. \u00a71681s-2(b)."
        result = label_claim_types(text)
        assert "1681s-2" in result

    def test_no_sections_found(self):
        text = "This is a procedural order about scheduling."
        result = label_claim_types(text)
        assert result == []

    def test_usc_format(self):
        text = "15 U.S.C. \u00a7 1692g requires debt validation."
        result = label_claim_types(text)
        assert "1692g" in result

    def test_text_format_section(self):
        text = "section 1692d of the Fair Debt Collection Practices Act"
        result = label_claim_types(text)
        assert "1692d" in result

    def test_deduplication(self):
        text = "\u00a71692e was violated. The \u00a71692e violation is clear. Under section 1692e the defendant..."
        result = label_claim_types(text)
        assert result.count("1692e") == 1
