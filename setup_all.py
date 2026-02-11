import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
VENV_PY = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd, cwd=None):
    print(f"\n> {cmd}")
    subprocess.check_call(cmd, cwd=cwd or ROOT, shell=True)


def find_python():
    # Prefer current interpreter if it is a regular Python install
    if sys.executable and Path(sys.executable).exists():
        return sys.executable
    # Fallback to PATH
    py = shutil.which("python")
    if py:
        return py
    raise RuntimeError("Python not found on PATH. Install Python 3.10+ and retry.")


def ensure_venv():
    if not VENV_PY.exists():
        py = find_python()
        run(f'"{py}" -m venv "{VENV_DIR}"')
    return VENV_PY


def upgrade_pip(venv_python):
    run(f'"{venv_python}" -m pip install --upgrade pip setuptools wheel')


def install_backend(venv_python):
    requirements = ROOT / "requirements.txt"
    if not requirements.exists():
        raise RuntimeError("requirements.txt not found in project root.")
    run(f'"{venv_python}" -m pip install -r "{requirements}"')


def install_frontend():
    if not (ROOT / "package.json").exists():
        raise RuntimeError("package.json not found in project root.")
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("npm not found on PATH. Install Node.js and retry.")
    run("npm install")


def run_verify(venv_python):
    verify = ROOT / "setup_verify.py"
    if verify.exists():
        run(f'"{venv_python}" "{verify}"')


def warn_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        print("\nWARNING: .env file is missing. Create it before running backend.")


def main():
    print(f"Project root: {ROOT}")
    venv_python = ensure_venv()
    upgrade_pip(venv_python)
    install_backend(venv_python)
    install_frontend()
    run_verify(venv_python)
    warn_env()

    print("\nSetup complete.")
    print("\nTo start backend:")
    print(f"  \"{venv_python}\" \"{ROOT / 'backend' / 'main.py'}\"")
    print("\nTo start frontend (new terminal):")
    print("  npm run dev")


if __name__ == "__main__":
    main()
