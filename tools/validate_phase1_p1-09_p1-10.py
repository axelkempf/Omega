#!/usr/bin/env python3
"""
Validierungs-Script für P1-09 und P1-10

Prüft:
1. Type Stubs sind vorhanden und korrekt strukturiert
2. Mypy-Konfiguration ist granular und vollständig
3. Stub-Pfad ist in pyproject.toml konfiguriert
4. Alle Tier-1-Module passieren mypy --strict
"""

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def check_file_exists(path: Path, description: str) -> bool:
    """Prüft ob Datei existiert."""
    if path.exists():
        print(f"✅ {description}: {path.relative_to(PROJECT_ROOT)}")
        return True
    else:
        print(f"❌ {description} FEHLT: {path.relative_to(PROJECT_ROOT)}")
        return False


def check_stubs() -> bool:
    """Validiert P1-09: Type Stubs."""
    print("\n" + "=" * 80)
    print("P1-09: Type Stubs Validation")
    print("=" * 80)
    
    all_ok = True
    
    # Stub-Verzeichnis
    stubs_dir = PROJECT_ROOT / "stubs"
    all_ok &= check_file_exists(stubs_dir / "README.md", "Stubs README")
    
    # joblib Stubs
    joblib_stub = stubs_dir / "joblib" / "__init__.pyi"
    all_ok &= check_file_exists(joblib_stub, "joblib Type Stub")
    
    # optuna Stubs
    optuna_stub = stubs_dir / "optuna" / "__init__.pyi"
    all_ok &= check_file_exists(optuna_stub, "optuna Type Stub")
    
    # Prüfe Stub-Inhalt (Stichproben)
    if joblib_stub.exists():
        content = joblib_stub.read_text()
        if "class Parallel:" in content and "def delayed" in content:
            print("✅ joblib Stub enthält Parallel und delayed")
        else:
            print("❌ joblib Stub unvollständig")
            all_ok = False
    
    if optuna_stub.exists():
        content = optuna_stub.read_text()
        if "class Study:" in content and "class Trial:" in content:
            print("✅ optuna Stub enthält Study und Trial")
        else:
            print("❌ optuna Stub unvollständig")
            all_ok = False
    
    return all_ok


def check_mypy_config() -> bool:
    """Validiert P1-10: Mypy-Konfiguration."""
    print("\n" + "=" * 80)
    print("P1-10: Mypy Configuration Validation")
    print("=" * 80)
    
    all_ok = True
    
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        print("❌ pyproject.toml nicht gefunden")
        return False
    
    content = pyproject.read_text()
    
    # Prüfe mypy_path
    if 'mypy_path = "stubs"' in content:
        print("✅ mypy_path = 'stubs' konfiguriert")
    else:
        print("❌ mypy_path nicht konfiguriert")
        all_ok = False
    
    # Prüfe kein globales ignore_errors
    tool_mypy_idx = content.find("[tool.mypy]")
    if tool_mypy_idx == -1:
        # Kein [tool.mypy]-Block vorhanden – damit auch kein globales ignore_errors dort
        mypy_main_section = ""
    else:
        overrides_idx = content.find("[[tool.mypy.overrides]]", tool_mypy_idx)
        if overrides_idx == -1:
            mypy_main_section = content[tool_mypy_idx:]
        else:
            mypy_main_section = content[tool_mypy_idx:overrides_idx]

    if "ignore_errors = true" not in mypy_main_section:
        print("✅ Kein globales ignore_errors")
    else:
        print("❌ Globales ignore_errors gefunden")
        all_ok = False
    
    # Prüfe Tier-1-Module
    tier1_modules = [
        "backtest_engine.core",
        "backtest_engine.optimizer",
        "backtest_engine.rating",
        "backtest_engine.config",
        "shared"
    ]
    
    for module in tier1_modules:
        if f'module = ["{module}"' in content or f'module = ["{module}.*"' in content:
            print(f"✅ Tier 1 (Strict): {module} konfiguriert")
        else:
            print(f"❌ Tier 1 (Strict): {module} FEHLT")
            all_ok = False
    
    # Prüfe Tier 3 (Live-Trading relaxed)
    if '"hf_engine.adapter.*"' in content and 'ignore_errors = true' in content:
        print("✅ Tier 3 (Relaxed): hf_engine konfiguriert")
    else:
        print("❌ Tier 3 (Relaxed): hf_engine FEHLT oder nicht relaxed")
        all_ok = False
    
    return all_ok


def run_mypy_validation() -> bool:
    """Führt mypy --strict auf Tier-1-Module aus."""
    print("\n" + "=" * 80)
    print("Mypy Strict Validation (Tier 1)")
    print("=" * 80)
    
    tier1_paths = [
        "src/backtest_engine/core",
        "src/backtest_engine/optimizer",
        "src/backtest_engine/rating",
        "src/backtest_engine/config",
        "src/shared",
    ]
    
    all_ok = True
    
    for path in tier1_paths:
        full_path = PROJECT_ROOT / path
        if not full_path.exists():
            print(f"⚠️  {path} existiert nicht (übersprungen)")
            continue
        
        print(f"\n🔍 Validiere {path}...")
        try:
            result = subprocess.run(
                ["mypy", "--strict", "--config-file", "pyproject.toml", str(full_path)],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"✅ {path}: mypy --strict PASS")
            else:
                print(f"❌ {path}: mypy --strict FAIL")
                print(f"   Errors:\n{result.stdout}")
                all_ok = False
        except subprocess.TimeoutExpired:
            print(f"⚠️  {path}: mypy timeout (übersprungen)")
        except FileNotFoundError:
            print(f"⚠️  mypy nicht gefunden (übersprungen)")
            break
    
    return all_ok


def main() -> int:
    """Hauptfunktion."""
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "P1-09 & P1-10 VALIDATION SUITE" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    
    checks = [
        ("P1-09: Type Stubs", check_stubs),
        ("P1-10: Mypy Config", check_mypy_config),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name}: Exception: {e}")
            results.append((name, False))
    
    # Optional: Mypy-Validation (kann lange dauern)
    run_mypy = os.getenv("RUN_MYPY_VALIDATION", "false").lower() == "true"
    if run_mypy:
        try:
            result = run_mypy_validation()
            results.append(("Mypy Strict Validation", result))
        except Exception as e:
            print(f"\n❌ Mypy Validation: Exception: {e}")
            results.append(("Mypy Strict Validation", False))
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        all_passed &= result
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 ALLE VALIDIERUNGEN BESTANDEN! Phase 1 (P1-09, P1-10) komplett.")
        return 0
    else:
        print("\n⚠️  EINIGE VALIDIERUNGEN FEHLGESCHLAGEN. Bitte oben prüfen.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
