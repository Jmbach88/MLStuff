import os
import pytest

os.environ["ML_LOCAL_DB"] = ":memory:"

from label import label_outcome


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
