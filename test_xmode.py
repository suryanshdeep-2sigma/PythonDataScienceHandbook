# test_magic.py
from alps.ipython.magic import line_magic, _handler

# 1. Set mode to PLAIN
print("--- Testing PLAIN Mode ---")
line_magic('xmode', 'context')

try:
    def buggy(): return 1/0
    buggy()
except Exception:
    # Simulate what Pyodide does: it catches the error and formats it
    import traceback
    # This calls our patched version!
    traceback.print_exc()

# 2. Set mode to VERBOSE
print("\n--- Testing VERBOSE Mode ---")
line_magic('xmode', 'verbose')

try:
    def deep_crash(x):
        local_var = "I should be visible!"
        return 10 / x
    deep_crash(0)
except Exception:
    import traceback
    traceback.print_exc()