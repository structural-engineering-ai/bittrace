# tests/test_main.py
def test_main_runs_without_error():
    import subprocess
    result = subprocess.run(["python", "main.py"], capture_output=True, timeout=60)
    assert result.returncode == 0
