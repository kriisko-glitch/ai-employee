"""CLI scaffolding — `aie init` writes a correctly-templated agent.yaml.

Regression: the original copytree-then-rewrite path was guarded by a
`not exists` check that always evaluated false, so the rewrite was skipped.
"""
import argparse
import os
from pathlib import Path

from ai_employee.cli import cmd_init


def _build_repo(tmp_path: Path) -> Path:
    """Build a minimal repo with an agents/example skeleton and an
    agent.yaml.example template."""
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "agents" / "example").mkdir(parents=True)
    (tmp_path / "agents" / "example" / "SOUL.md").write_text("example soul")
    (tmp_path / "agents" / "example" / "MEMORY.md").write_text("example memory")
    (tmp_path / "agents" / "example" / "agent.yaml").write_text(
        "name: example\nworkspace: agents/example\n"
    )
    (tmp_path / "agent.yaml.example").write_text(
        "name: example\nworkspace: agents/example\nmodel:\n  provider: openai\n"
    )
    return tmp_path


def test_init_substitutes_name_and_workspace(tmp_path: Path, monkeypatch):
    repo = _build_repo(tmp_path)
    monkeypatch.setenv("AIE_HOME", str(repo))

    cmd_init(argparse.Namespace(name="bob"))

    bob_dir = repo / "agents" / "bob"
    assert bob_dir.is_dir()
    # SOUL/MEMORY should have been copied from the example.
    assert (bob_dir / "SOUL.md").read_text() == "example soul"
    # agent.yaml must have name/workspace rewritten — not left as 'example'.
    rendered = (bob_dir / "agent.yaml").read_text()
    assert "name: bob" in rendered
    assert "workspace: agents/bob" in rendered
    assert "name: example" not in rendered
    assert "workspace: agents/example" not in rendered


def test_init_skips_runtime_files(tmp_path: Path, monkeypatch):
    """state.json / memory.db / drafts/ from the example must not be copied."""
    repo = _build_repo(tmp_path)
    (repo / "agents" / "example" / "state.json").write_text('{"ghost": true}')
    (repo / "agents" / "example" / "memory.db").write_text("not a real db")
    (repo / "agents" / "example" / "drafts").mkdir()
    (repo / "agents" / "example" / "drafts" / "ghost.md").write_text("old")

    monkeypatch.setenv("AIE_HOME", str(repo))
    cmd_init(argparse.Namespace(name="alice"))

    alice = repo / "agents" / "alice"
    assert not (alice / "state.json").exists()
    assert not (alice / "memory.db").exists()
    assert not (alice / "drafts").exists()


def test_init_refuses_to_clobber(tmp_path: Path, monkeypatch, capsys):
    repo = _build_repo(tmp_path)
    (repo / "agents" / "existing").mkdir()
    monkeypatch.setenv("AIE_HOME", str(repo))

    rc = cmd_init(argparse.Namespace(name="existing"))
    assert rc == 1
    captured = capsys.readouterr()
    assert "already exists" in captured.err
