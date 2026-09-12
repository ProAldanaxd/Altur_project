import pytest


@pytest.fixture(autouse=True)
def isolated_audit_database(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIT_DB_PATH", str(tmp_path / "audit.sqlite3"))
