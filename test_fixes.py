#!/usr/bin/env python3
"""
Quick test script to verify the surgical patches are working
"""

import sys
import logging
from pathlib import Path

# Test 1: Verify logger crash is fixed in regenerate_policy_data.py
def test_logger_fix():
    print("🔍 Test 1: Logger initialization fix...")
    
    # Import the fixed script - this should not crash even if PyPDF2 is missing
    sys.path.insert(0, str(Path(__file__).parent / "scripts" / "mistral"))
    
    try:
        import regenerate_policy_data
        print("✅ regenerate_policy_data.py imports successfully (logger defined before use)")
        return True
    except NameError as e:
        if "logger" in str(e):
            print(f"❌ Logger crash still present: {e}")
            return False
        else:
            raise e
    except Exception as e:
        print(f"⚠️  Import failed for other reason: {e}")
        return False


# Test 2: Verify training script imports without HeartbeatManager dependency
def test_heartbeat_import_fix():
    print("🔍 Test 2: Optional heartbeat imports fix...")
    
    try:
        import train_mistral_simple
        print("✅ train_mistral_simple.py imports successfully (heartbeat imports are optional)")
        return True
    except ImportError as e:
        if "heartbeat" in str(e).lower():
            print(f"❌ Heartbeat import crash still present: {e}")
            return False
        else:
            raise e
    except Exception as e:
        print(f"⚠️  Import failed for other reason: {e}")
        return False


# Test 3: Verify monitoring script accepts correct arguments
def test_monitoring_args():
    print("🔍 Test 3: Monitor script argument compatibility...")
    
    import subprocess
    monitor_script = Path(__file__).parent / "scripts" / "monitoring" / "monitor.py"
    
    if not monitor_script.exists():
        print(f"❌ Monitor script not found at {monitor_script}")
        return False
    
    # Test that --watch is rejected (as expected)
    result = subprocess.run([
        "python3", str(monitor_script), 
        "--run-id", "test-run",
        "--once",  # Use --once for quick test
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ monitor.py runs with correct arguments")
        return True
    else:
        print(f"❌ monitor.py failed with correct args: {result.stderr}")
        return False


# Test 4: Verify production monitor wrapper works
def test_production_monitor():
    print("🔍 Test 4: Production monitor wrapper fix...")
    
    wrapper_script = Path(__file__).parent / "scripts" / "monitoring" / "production_monitor.sh"
    
    if not wrapper_script.exists():
        print(f"❌ Production monitor script not found at {wrapper_script}")
        return False
    
    # Check that --watch is no longer in default args
    with open(wrapper_script, 'r') as f:
        content = f.read()
    
    if '--watch --interval 60' in content:
        print("❌ production_monitor.sh still contains --watch in default args")
        return False
    elif '--interval 60' in content and '--watch' not in content:
        print("✅ production_monitor.sh fixed (no --watch in default args)")
        return True
    else:
        print("⚠️  production_monitor.sh default args unclear")
        return False


def main():
    print("=" * 60)
    print("🧪 Testing surgical patches for training pipeline fixes")
    print("=" * 60)
    
    tests = [
        ("Logger crash fix", test_logger_fix),
        ("Heartbeat import fix", test_heartbeat_import_fix), 
        ("Monitor args fix", test_monitoring_args),
        ("Production monitor fix", test_production_monitor),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    print("=" * 60)
    print("📊 Test Summary:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🚀 All fixes verified! Pipeline should now work.")
        return 0
    else:
        print("⚠️  Some issues remain - check failed tests above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())