"""Run all tests for EM-Refinement Loop."""

import sys
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

if __name__ == "__main__":
    print("=" * 70)
    print("EM-Refinement Loop Test Suite")
    print("=" * 70 + "\n")
    
    # Run tests in order
    test_modules = [
        ("Model Tests", "em_loop.tests.test_models"),
        ("Loss Function Tests", "em_loop.tests.test_losses"),
        ("EM Loop Tests", "em_loop.tests.test_em_loop"),
        ("Integration Tests", "em_loop.tests.test_integration"),
    ]
    
    results = []
    
    for test_name, module_name in test_modules:
        print(f"\n{'='*70}")
        print(f"Running {test_name}...")
        print(f"{'='*70}\n")
        
        try:
            # Run the test module as a subprocess
            result = subprocess.run(
                [sys.executable, "-m", module_name],
                capture_output=False,
                text=True,
            )
            
            if result.returncode == 0:
                results.append((test_name, True, None))
            else:
                results.append((test_name, False, f"Exit code: {result.returncode}"))
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed, error in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {test_name}")
        if error:
            print(f"    Error: {error}")
        all_passed = all_passed and passed
    
    print("=" * 70)
    
    if all_passed:
        print("All tests passed! ✓")
        sys.exit(0)
    else:
        print("Some tests failed. See details above.")
        sys.exit(1)
