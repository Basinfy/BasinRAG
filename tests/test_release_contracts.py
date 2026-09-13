from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_image_does_not_copy_workspace_or_local_data():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert "COPY . ." not in dockerfile
    assert "COPY basinrag ./basinrag" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    for excluded in (".env", ".basinrag", "results", "logs", ".mcp.json", "graphify-out"):
        assert excluded in dockerignore


def test_public_python_snippet_and_version_are_current():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "from basinrag import BasinRAG, BasinRAGConfig" in readme
    assert 'storage_dir=".basinrag-v3"' in readme
    assert "version      = {v1.1.0}" in readme


def test_repository_has_no_approved_quantitative_release_claims():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    comparison = (ROOT / "docs" / "BENCHMARK_COMPARISON_pt.md").read_text(encoding="utf-8")
    index = (ROOT / "docs" / "INDEX.md").read_text(encoding="utf-8")
    assert "0.445" not in changelog
    assert "0.733" not in comparison
    assert "BENCHMARK_COMPARISON_pt.md" not in index
