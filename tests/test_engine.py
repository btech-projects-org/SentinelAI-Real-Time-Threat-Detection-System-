
import pytest
from backend.services.dl_threat_engine import DeepLearningThreatEngine

@pytest.fixture
def engine():
    return DeepLearningThreatEngine()

def test_fighting_heuristic_no_overlap(engine):
    """two people far apart should not trigger fighting"""
    persons = [
        {"box": [0.0, 0.0, 0.1, 0.1], "confidence": 0.9}, # Top Left
        {"box": [0.5, 0.5, 0.1, 0.1], "confidence": 0.9}  # Center
    ]
    events = engine.detect_fighting(persons)
    assert len(events) == 0

def test_fighting_heuristic_overlap(engine):
    """two people overlapping significantly should trigger fighting"""
    persons = [
        {"box": [0.4, 0.4, 0.2, 0.2], "confidence": 0.9},
        {"box": [0.45, 0.45, 0.2, 0.2], "confidence": 0.9} # High Overlap
    ]
    events = engine.detect_fighting(persons)
    assert len(events) == 1
    assert events[0]["label"] == "VIOLENCE"
    assert events[0]["metadata"]["severity"] == "CRITICAL"

def test_fighting_heuristic_minor_overlap(engine):
    """minor overlap should not trigger"""
    persons = [
        {"box": [0.0, 0.0, 0.2, 0.2], "confidence": 0.9},
        {"box": [0.19, 0.0, 0.2, 0.2], "confidence": 0.9} # 0.01 width overlap
    ]
    events = engine.detect_fighting(persons)
    assert len(events) == 0

def test_weapon_logic_parsing(engine):
    """Test standard weapon class set"""
    assert 'knife' in engine.weapon_classes
    assert 'gun' in engine.weapon_classes
