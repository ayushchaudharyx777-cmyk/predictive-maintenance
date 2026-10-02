from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_renders(dirs):
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Predictive Maintenance"
    assert len(at.metric) >= 5
