import os
import pytest
from unittest.mock import patch, MagicMock
import json

os.environ["ML_LOCAL_DB"] = ":memory:"

from label import label_outcome, label_claim_types
from db import get_local_engine, init_local_db, get_session, Label, Opinion
from sqlalchemy import text


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


class TestExpandedOutcomePatterns:
    def test_motion_to_dismiss_granted(self):
        text = "Defendant's motion to dismiss is hereby granted."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"

    def test_motion_to_dismiss_denied(self):
        text = "The motion to dismiss filed by defendant is denied."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_order_granting_motion_to_dismiss(self):
        text = "ORDER GRANTING DEFENDANT'S MOTION TO DISMISS."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"

    def test_order_denying_motion_to_dismiss(self):
        text = "ORDER DENYING MOTION TO DISMISS."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_default_judgment(self):
        text = "Default judgment is entered against the defendant in the amount of $5,000."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_affirmed_low_weight(self):
        text = "The judgment of the district court is AFFIRMED."
        result = label_outcome(text)
        assert result["label"] == "defendant_win"

    def test_reversed_low_weight(self):
        text = "The district court's order is REVERSED."
        result = label_outcome(text)
        assert result["label"] == "plaintiff_win"

    def test_affirmed_in_part_reversed_in_part(self):
        text = "The judgment is AFFIRMED IN PART and REVERSED IN PART."
        result = label_outcome(text)
        assert result["label"] == "mixed"

    def test_vacated_and_remanded(self):
        text = "The order is VACATED and REMANDED for further proceedings."
        result = label_outcome(text)
        assert result["label"] == "mixed"

    def test_settled(self):
        text = "The parties have reached a settlement. The case is dismissed pursuant to stipulation."
        result = label_outcome(text)
        assert result["label"] == "settled"

    def test_voluntary_dismissal(self):
        text = "Pursuant to the parties' stipulation of voluntary dismissal, this case is closed."
        result = label_outcome(text)
        assert result["label"] == "settled"


class TestBatchLabeling:
    def _setup_opinions(self):
        engine = get_local_engine()
        init_local_db(engine)
        session = get_session(engine)

        session.execute(text("DELETE FROM labels"))
        session.execute(text("DELETE FROM opinions"))
        session.commit()

        opinions = [
            Opinion(id=1, package_id="p1", title="Smith v. Collector",
                    plain_text="Judgment for plaintiff. Damages awarded of $1000 under \u00a71692e."),
            Opinion(id=2, package_id="p2", title="Jones v. Agency",
                    plain_text="Plaintiff's complaint is dismissed. Judgment for defendant."),
            Opinion(id=3, package_id="p3", title="Brown v. Corp",
                    plain_text="This is a scheduling order for the pretrial conference."),
        ]
        session.add_all(opinions)
        session.commit()
        return engine

    def test_run_labeling_creates_labels(self):
        from label import run_labeling
        engine = self._setup_opinions()
        stats = run_labeling(engine)
        session = get_session(engine)

        outcome_labels = session.query(Label).filter_by(label_type="outcome").all()
        labeled = [l for l in outcome_labels if l.label_value != "unlabeled"]
        assert len(labeled) >= 2
        session.close()

    def test_run_labeling_creates_claim_labels(self):
        from label import run_labeling
        engine = self._setup_opinions()
        stats = run_labeling(engine)
        session = get_session(engine)

        claim_labels = session.query(Label).filter_by(label_type="claim_type").all()
        values = [l.label_value for l in claim_labels]
        assert "1692e" in values
        session.close()

    def test_run_labeling_returns_stats(self):
        from label import run_labeling
        engine = self._setup_opinions()
        stats = run_labeling(engine)
        assert "outcome" in stats
        assert "claim_type" in stats
        assert stats["outcome"]["total"] == 3


class TestLLMLabeling:
    def test_parse_llm_response_valid(self):
        from label import _parse_llm_response
        response = '{"outcome": "plaintiff_win", "confidence": 0.85}'
        result = _parse_llm_response(response)
        assert result["label"] == "plaintiff_win"
        assert result["confidence"] == 0.85

    def test_parse_llm_response_with_text_around_json(self):
        from label import _parse_llm_response
        response = 'Based on my analysis:\n{"outcome": "defendant_win", "confidence": 0.9}\nThat is my conclusion.'
        result = _parse_llm_response(response)
        assert result["label"] == "defendant_win"

    def test_parse_llm_response_invalid(self):
        from label import _parse_llm_response
        response = "I cannot determine the outcome of this case."
        result = _parse_llm_response(response)
        assert result["label"] == "unclear"
        assert result["confidence"] == 0.0

    def test_parse_llm_response_unclear(self):
        from label import _parse_llm_response
        response = '{"outcome": "unclear", "confidence": 0.3}'
        result = _parse_llm_response(response)
        assert result["label"] == "unclear"

    def test_parse_llm_response_settled(self):
        from label import _parse_llm_response
        response = '{"outcome": "settled", "confidence": 0.8}'
        result = _parse_llm_response(response)
        assert result["label"] == "settled"

    @patch("label.requests.post")
    def test_label_with_llm_success(self, mock_post):
        from label import label_with_llm
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"outcome": "plaintiff_win", "confidence": 0.9}'
        }
        mock_post.return_value = mock_response
        result = label_with_llm("The court grants judgment for plaintiff.")
        assert result["label"] == "plaintiff_win"
        assert result["confidence"] == 0.9

    @patch("label.requests.post")
    def test_label_with_llm_connection_error(self, mock_post):
        from label import label_with_llm
        import requests as req
        mock_post.side_effect = req.ConnectionError("Ollama not running")
        result = label_with_llm("Some opinion text")
        assert result["label"] == "error"
        assert result["confidence"] == 0.0
