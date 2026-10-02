from pathlib import Path

import pytest

from form_sequencer.config import Project
from form_sequencer.fixture import make_fixture
from form_sequencer.pipeline import analyze_audio_step, analyze_documents_step


def _quiet(msg: str) -> None:
    pass


@pytest.fixture(scope="session")
def fixture_project(tmp_path_factory) -> Project:
    """A synthetic project, analyzed once for the whole test session."""
    p = Project(Path(tmp_path_factory.mktemp("fixture")))
    make_fixture(p, seed=1)
    settings = p.load_settings()
    analyze_audio_step(p, settings, _quiet)
    analyze_documents_step(p, settings, _quiet)
    from form_sequencer import stamps
    stamps.process(p, settings, _quiet)
    stamps.process(p, settings, _quiet, "writing")
    return p
