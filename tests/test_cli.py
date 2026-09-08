import subprocess
import sys

def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "basinrag.cli", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout

def test_cli_query_without_index(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "basinrag.cli", "query", "teste"],
        capture_output=True, text=True, cwd=str(tmp_path)
    )
    assert result.returncode != 0 or "Nenhum indice" in result.stdout
