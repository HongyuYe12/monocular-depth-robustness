import pytest

from depth_wild.subject import SubjectCandidate, box_iou, suppress_duplicate_candidates


def test_box_iou_identical_and_separate():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_suppress_duplicate_candidates_keeps_high_confidence_same_class():
    candidates = [
        SubjectCandidate("old-1", "couch", 0.70, (0, 0, 100, 100)),
        SubjectCandidate("old-2", "couch", 0.90, (5, 5, 100, 100)),
        SubjectCandidate("old-3", "person", 0.80, (5, 5, 100, 100)),
    ]

    kept = suppress_duplicate_candidates(candidates, iou_threshold=0.55)

    # 两个 couch 高度重叠时保留置信度更高者；不同类别的 person 仍然保留。
    assert [(item.subject_id, item.label, item.confidence) for item in kept] == [
        ("subject-1", "couch", 0.90),
        ("subject-2", "person", 0.80),
    ]


def test_suppress_duplicate_candidates_validates_threshold():
    with pytest.raises(ValueError):
        suppress_duplicate_candidates([], iou_threshold=1.1)
