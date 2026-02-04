"""
Robust IPython Magic Commands Implementation for Pyodide

This module provides Python implementations of IPython magic commands.
Place at: /textbook/alps/ipython/magics.py

Features:
- Proper argument parsing using shlex and argparse
- Correct namespace access for timeit and exec
- Pythonic implementation using standard library callables
- Handles nested loops, complex expressions, and edge cases
- Supports async operations (pip) and debugging (pdb)
"""

import os
import sys
import time
import timeit
import runpy
import traceback
import shlex
import re
import statistics
import pdb
import io
import cProfile
import pstats
from typing import Any, Dict, Optional, Tuple, List

class MagicHandler:
    """Handles execution of magic commands with proper namespace management"""
    
    def __init__(self):
        self.last_execution_time = None
        self.xmode = 'context' 
        
        self._dh = [os.getcwd()]  # Directory history stack
        self._bookmarks = {}      # Bookmarks dictionary
        self.aliases = {}         # for storing the aliases for the magic commands

        # Save original traceback functions
        self._orig_format_exception = traceback.format_exception
        self._orig_print_exception = traceback.print_exception
        
        # Monkeypatch traceback module
        # This ensures that ANYTIME an error is printed,
        # it goes through our logic to support %xmode
        traceback.format_exception = self._custom_format_exception
        traceback.print_exception = self._custom_print_exception
        
    def _get_caller_namespace(self) -> Dict[str, Any]:
        """
        Get the namespace where the magic was called from.
        Goes up the stack to find the actual user namespace.
        """
        # Go up 4 frames to bypass the MagicHandler and wrapper functions
        # 0 = _get_caller_namespace
        # 1 = _magic_* method
        # 2 = MagicHandler.line_magic/cell_magic
        # 3 = Global line_magic/cell_magic wrapper
        # 4 = The Notebook Cell (Where 'L' lives!)
        try:
            frame = sys._getframe(4)
        except ValueError:
            # Fallback for direct calls or shallower stacks
            frame = sys._getframe(3)

        # Merge globals and locals (Locals must overwrite globals)
        ns = frame.f_globals.copy()
        ns.update(frame.f_locals)
        return ns
        
    def line_magic(self, command: str, args: str) -> Any:
        """Execute a line magic command"""
        # check aliases first
        if command in self.aliases:
            command = self.aliases[command]

        method_name = f"_magic_{command}"
        
        if hasattr(self, method_name):
            # RETURN the result (Crucial for async pip)
            return getattr(self, method_name)(args)
        else:
            print(f"UsageError: Line magic function `%{command}` not found.", file=sys.stderr)
            return None
    
    def cell_magic(self, command: str, args: str) -> Any:
        """Execute a cell magic command"""
        # check aliases first
        if command in self.aliases:
            command = self.aliases[command]
        
        method_name = f"_cell_magic_{command}"
        
        if hasattr(self, method_name):
            # RETURN the result
            return getattr(self, method_name)(args)
        else:
            print(f"UsageError: Cell magic function `%%{command}` not found.", file=sys.stderr)
            return None
            
    # ============ EXCEPTION HANDLING ============
    def _normalize_exception_args(self, exc, value=None, tb=None):
        """Helper to handle Python 3.10+ single-argument exception calls"""
        if isinstance(exc, type):
            # Old style: (type, value, traceback)
            return exc, value, tb
        else:
            # New style: (ExceptionInstance)
            # We derive type and traceback from the instance
            return type(exc), exc, exc.__traceback__

    def _custom_format_exception(self, exc, value=None, tb=None, *args, **kwargs):
        """Custom traceback formatter that respects %xmode"""
        # Normalize arguments to ensure we always have (etype, value, tb)
        etype, value, tb = self._normalize_exception_args(exc, value, tb)

        # Save crash data for %debug
        sys.last_type = etype
        sys.last_value = value
        sys.last_traceback = tb

        if self.xmode == 'plain':
            # Just the error message, no stack trace
            return traceback.format_exception_only(etype, value)
            
        elif self.xmode == 'verbose':
            # Detailed stack trace with local variable values
            try:
                # capture_locals=True shows variable values!
                tb_exc = traceback.TracebackException(etype, value, tb, capture_locals=True)
                return list(tb_exc.format())
            except Exception:
                # Fallback if something fails
                return self._orig_format_exception(etype, value, tb, *args, **kwargs)
                
        else: # 'context' (Default)
            return self._orig_format_exception(etype, value, tb, *args, **kwargs)

    def _custom_print_exception(self, exc, value=None, tb=None, limit=None, file=None, chain=True):
        """Custom print_exception that uses our formatter"""
        # Normalize arguments
        etype, value, tb = self._normalize_exception_args(exc, value, tb)
        
        f = file or sys.stderr
        # Pass normalized args to our custom formatter
        for line in self._custom_format_exception(etype, value, tb, limit=limit, chain=chain):
            print(line, end='', file=f)

    def _magic_xmode(self, args: str) -> None:
        """Set exception reporting mode: Plain, Context, or Verbose."""
        mode = args.strip().lower()
        valid_modes = ['plain', 'context', 'verbose']
        
        if mode in valid_modes:
            self.xmode = mode
            print(f"Exception reporting mode: {mode}")
        elif not mode:
            print(f"Current exception mode: {self.xmode}")
        else:
            print(f"Error: Invalid mode '{mode}'. Valid modes: {', '.join(valid_modes)}", file=sys.stderr)

    def _magic_debug(self, args: str) -> None:
        """Activate the interactive debugger (pdb)."""
        # Check if we have a traceback saved from a previous crash
        if not hasattr(sys, 'last_traceback') or sys.last_traceback is None:
            print("No traceback has been saved. Run code that crashes first.", file=sys.stderr)
            return

        print("Entering interactive debugger (pdb). Type 'q' to quit.")
        pdb.post_mortem(sys.last_traceback)

    # ============ HELPERS ============

    def _format_time(self, timespan: float, precision: int = 2) -> str:
        """Formats a time duration with appropriate units (ns, µs, ms, s)."""
        if timespan >= 1:
            return f"{timespan:.{precision}f} s"
        elif timespan >= 1e-3:
            return f"{timespan * 1e3:.{precision}f} ms"
        elif timespan >= 1e-6:
            return f"{timespan * 1e6:.{precision}f} µs"
        else:
            return f"{timespan * 1e9:.{precision}f} ns"
        
    def _run_with_profiler(self, code: str, namespace: dict) -> None:
        """Helper to run code with cProfile and print stats"""
        prof = cProfile.Profile()
        try:
            prof.runctx(code, namespace, namespace)
            s = io.StringIO()
            # Sort by cumulative time by default
            ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
            ps.print_stats()
            print(s.getvalue())
        except Exception:
            traceback.print_exc()
    # ============ LINE MAGICS ============

    def _magic_lsmagic(self, args: str) -> None:
        """List currently available magic functions."""
        line_magics = sorted([m.replace('_magic_', '%') for m in dir(self) if m.startswith('_magic_')])
        cell_magics = sorted([m.replace('_cell_magic_', '%%') for m in dir(self) if m.startswith('_cell_magic_')])
        
        print("Available line magics:")
        print("  " + "  ".join(line_magics))
        print("\nAvailable cell magics:")
        print("  " + "  ".join(cell_magics))
    
    def _magic_history(self, args: str) -> None:
        """
        Print input history (_i<n> variables).
        Usage:
          %history          - list all history
          %history -l 5     - list last 5 commands
          %history -n 10-15 - list commands from index 10 to 15
          %history 5-10     - implicit range (same as -n)
        """
        ns = self._get_caller_namespace()
        
        # Try to find the history list
        history = ns.get('In') or ns.get('_ih')
        if not isinstance(history, list):
            print("History not found. (Ensure execution loop populates 'In' or '_ih')")
            return

        # Default: Show everything
        start_idx = 0
        end_idx = len(history)
        
        # Parse arguments
        tokens = args.strip().split()
        
        try:
            if '-l' in tokens:
                # CASE: Last N lines (%history -l 3)
                idx = tokens.index('-l')
                if idx + 1 < len(tokens):
                    count = int(tokens[idx+1])
                    start_idx = max(0, len(history) - count)
                    end_idx = len(history)
            
            elif '-n' in tokens:
                # CASE: Specific Range (%history -n 1-5)
                idx = tokens.index('-n')
                if idx + 1 < len(tokens):
                    range_str = tokens[idx+1]
                    if '-' in range_str:
                        s, e = map(int, range_str.split('-'))
                        start_idx = s
                        end_idx = e + 1  # Make it inclusive
                    else:
                        start_idx = int(range_str)
                        end_idx = start_idx + 1
            
            elif tokens:
                # CASE: Implicit Range (%history 1-5)
                first_arg = tokens[0]
                if '-' in first_arg:
                    s, e = map(int, first_arg.split('-'))
                    start_idx = s
                    end_idx = e + 1
                elif first_arg.isdigit():
                    start_idx = int(first_arg)
                    end_idx = start_idx + 1
                    
        except ValueError:
            print("Error: Invalid arguments. Use -l <n> or -n <start>-<end>", file=sys.stderr)
            return

        # Print the loop
        # We verify bounds to prevent crashing if user asks for index 1000
        for i in range(start_idx, end_idx):
            if 0 <= i < len(history):
                cmd = history[i]
                # Only print if not empty/whitespace
                if cmd.strip(): 
                    print(f"{i}: {cmd}")
    
    def _magic_who_ls(self, args: str) -> list[str]:
        """Return a sorted list of all interactive variables."""
        namespace = self._get_caller_namespace()
        user_vars = [k for k in namespace.keys() 
                     if not k.startswith('_') 
                     and k not in ['In', 'Out', '__builtins__', 'exit', 'quit']]
        return sorted(user_vars)
    
    def _magic_dhist(self, args: str) -> None:
        """Print your directory history."""
        for i, path in enumerate(self._dh):
            print(f"{i}: {path}")

    def _magic_bookmark(self, args: str) -> None:
        """
        Manage filesystem bookmarks.
        Usage: 
          %bookmark <name>       - set bookmark for current dir
          %bookmark <name> <dir> - set bookmark for specific dir
          %bookmark -l           - list all bookmarks
          %bookmark -d <name>    - delete bookmark
        """
        args_list = args.strip().split()
        
        if not args_list:
            print("Usage: %bookmark <name> [dir] | -l | -d <name>")
            return

        if args_list[0] == '-l':
            if not self._bookmarks:
                print("No bookmarks.")
            else:
                for name, path in self._bookmarks.items():
                    print(f"{name} -> {path}")
            return

        if args_list[0] == '-d':
            if len(args_list) < 2:
                print("Usage: %bookmark -d <name>")
            elif args_list[1] in self._bookmarks:
                del self._bookmarks[args_list[1]]
                print(f"Bookmark '{args_list[1]}' deleted.")
            else:
                print(f"Bookmark '{args_list[1]}' not found.")
            return

        # Set bookmark
        name = args_list[0]
        if len(args_list) > 1:
            target_dir = args_list[1]
        else:
            target_dir = os.getcwd()
            
        self._bookmarks[name] = target_dir
        print(f"Bookmark '{name}' set to {target_dir}")
    
    def _magic_pwd(self, args: str) -> str:
        """Print working directory"""
        cwd = os.getcwd()
        print(cwd)
        return
    
    def _magic_cd(self, args: str) -> None:
        """
        Change directory with history and bookmark support.
        Usage: cd [-q] <path> | - | -<n>
        """
        args_list = args.strip().split()
        path = ""
        quiet = False
        
        # Parse flags
        if '-q' in args_list:
            quiet = True
            args_list.remove('-q')
            
        if not args_list:
            path = os.path.expanduser("~")
        else:
            path = args_list[0]

        # 1. Handle History Jumps
        if path == '-':
            if len(self._dh) >= 2:
                path = self._dh[-2]
            else:
                print("Error: No previous directory.", file=sys.stderr); return
        elif path.startswith('-') and path[1:].isdigit():
            idx = int(path)
            if abs(idx) < len(self._dh):
                path = self._dh[idx]
            else:
                print(f"Error: History index {idx} out of range.", file=sys.stderr); return
        
        # 2. Handle Bookmarks
        elif path in self._bookmarks:
            path = self._bookmarks[path]
            if not quiet: print(f"(bookmark:{path})")

        # 3. Perform Change
        try:
            os.chdir(path)
            cwd = os.getcwd()
            
            # Update History (only if new)
            if not self._dh or self._dh[-1] != cwd:
                self._dh.append(cwd)
                
            if not quiet:
                print(cwd)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
    
    def _magic_ls(self, args: str) -> None:
        """List directory contents"""
        path = args.strip() if args.strip() else "."
        try:
            items = os.listdir(path)
            for item in sorted(items):
                print(item)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
    
    def _magic_who(self, args: str) -> None:
        """List variables in namespace"""
        namespace = self._get_caller_namespace()
        
        # Filter user variables
        user_vars = {
            k: v for k, v in namespace.items() 
            if not k.startswith('_') 
            and k not in ['In', 'Out', '__builtins__']
            and (not callable(v) or k in ['input', 'print'])
        }
        
        # Type filtering
        args_stripped = args.strip().lower()
        if args_stripped:
            type_map = {
                'str': str, 'int': int, 'float': float,
                'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            }
            if args_stripped in type_map:
                target_type = type_map[args_stripped]
                user_vars = {k: v for k, v in user_vars.items() if isinstance(v, target_type)}
        
        if user_vars:
            for var_name in sorted(user_vars.keys()):
                print(var_name, end='\t')
            print()
        else:
            print("Interactive namespace is empty.")
    
    def _magic_whos(self, args: str) -> None:
        """List variables with detailed information"""
        namespace = self._get_caller_namespace()
        
        user_vars = {
            k: v for k, v in namespace.items() 
            if not k.startswith('_') 
            and k not in ['In', 'Out', '__builtins__']
        }
        
        if user_vars:
            print(f"{'Variable':<20} {'Type':<20} {'Data/Info':<40}")
            print("-" * 80)
            
            for var_name in sorted(user_vars.keys()):
                var = user_vars[var_name]
                var_type = type(var).__name__
                try:
                    if isinstance(var, (list, tuple, set, dict)):
                        data_info = f"{var_type} (length {len(var)})"
                    elif isinstance(var, str):
                        data_info = repr(var)[:37] + "..." if len(var) > 40 else repr(var)
                    elif callable(var):
                        data_info = "<function>"
                    else:
                        data_info = str(var)[:40]
                except:
                    data_info = "<unprintable>"
                
                print(f"{var_name:<20} {var_type:<20} {data_info:<40}")
        else:
            print("Interactive namespace is empty.")
    
    def _magic_time(self, args: str) -> None:
        """Time execution of a single statement"""
        if not args.strip():
            print("Error: No statement to time", file=sys.stderr)
            return
        
        namespace = self._get_caller_namespace()
        
        start_time = time.perf_counter()
        start_process = time.process_time()
        
        try:
            exec(args, namespace)
        except Exception as e:
            traceback.print_exc()
            return
        finally:
            wall_time = time.perf_counter() - start_time
            cpu_time = time.process_time() - start_process
            
            cpu_disp = self._format_time(cpu_time)
            wall_disp = self._format_time(wall_time)

            print(f"CPU times: total: {cpu_disp}")
            print(f"Wall time: {wall_disp}")
            self.last_execution_time = wall_time
    
    def _parse_timeit_args(self, args: str) -> Tuple[int, int, str]:
        """Parse timeit arguments properly."""
        number = None
        repeat = 7
        try:
            tokens = shlex.split(args)
        except ValueError:
            tokens = args.split()
        
        stmt_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('-n'):
                try:
                    val = token[2:] if len(token) > 2 else tokens[i+1]
                    number = int(val)
                    if len(token) <= 2: i += 1
                except: pass
            elif token.startswith('-r'):
                try:
                    val = token[2:] if len(token) > 2 else tokens[i+1]
                    repeat = int(val)
                    if len(token) <= 2: i += 1
                except: pass
            else:
                stmt_tokens.append(token)
            i += 1
            
        stmt = ' '.join(stmt_tokens) if stmt_tokens else args
        if not stmt.strip(): stmt = args
        return number, repeat, stmt
    
    def _magic_timeit(self, args: str) -> None:
        """Time execution of a statement with multiple runs."""
        if not args.strip():
            print("Error: No statement to time", file=sys.stderr)
            return
        
        number, repeat, stmt = self._parse_timeit_args(args)
        namespace = self._get_caller_namespace()
        timer = timeit.Timer(stmt, globals=namespace)
        
        try:
            if number is None:
                for scale in [1, 10, 100, 1000, 10000, 100000, 1000000]:
                    if timer.timeit(scale) >= 0.2:
                        number = scale
                        break
                else: number = 1000000
            
            all_runs = timer.repeat(repeat=repeat, number=number)
            best = min(all_runs) / number
            stdev = statistics.stdev(all_runs) / number if len(all_runs) > 1 else 0
            
            self._print_timeit_result(best, stdev, repeat, number)
        except Exception as e:
            print(f"Error in timing: {e}", file=sys.stderr)
            traceback.print_exc()
    
    def _print_timeit_result(self, best: float, stdev: float, repeat: int, number: int) -> None:
        if best < 1e-6:
            val, sval, unit = best*1e9, stdev*1e9, "ns"
        elif best < 1e-3:
            val, sval, unit = best*1e6, stdev*1e6, "µs"
        elif best < 1:
            val, sval, unit = best*1e3, stdev*1e3, "ms"
        else:
            val, sval, unit = best, stdev, "s"
        
        print(f"{val:.2f} {unit} ± {sval:.2f} {unit} per loop "
              f"(mean ± std. dev. of {repeat} runs, {number} loops each)")

    def _magic_pip(self, args: str):
        """Install packages using micropip"""
        try:
            import micropip # type: ignore
        except ImportError:
            print("Error: micropip not available", file=sys.stderr)
            return None
        
        if 'install' in args:
            packages = args.replace('install', '').strip().split()
            if not packages:
                print("Error: No packages specified", file=sys.stderr)
                return None
            
            print(f"Installing {packages}...")
            # RETURN the coroutine so it can be awaited!
            return micropip.install(packages)
            
        elif 'list' in args:
            print("Listing installed packages...")
            try:
                import importlib.metadata
                dists = importlib.metadata.distributions()
                for dist in dists:
                    print(f"{dist.name} ({dist.version})")
            except:
                print("Unable to list packages")
        else:
            print(f"Usage: %pip install <package>")

    def _magic_run(self, args: str) -> None:
        """Run a Python file"""
        filename = args.strip()
        if not filename or not os.path.exists(filename):
            print(f"Error: File not found: {filename}", file=sys.stderr)
            return
        
        namespace = self._get_caller_namespace()
        try:
            with open(filename, 'r') as f:
                exec(f.read(), namespace)
        except Exception:
            traceback.print_exc()
    
    def _magic_prun(self, args: str) -> None:
        """Run a statement through the python code profiler."""
        if not args.strip():
            print("Usage: %prun <statement>")
            return
        self._run_with_profiler(args, self._get_caller_namespace())
    
    def _magic_reset_selective(self, args: str) -> None:
        """Clear names from namespace matching a regex."""
        regex = args.strip()
        if not regex:
            print("Usage: %reset_selective <regex>")
            return
        
        try:
            pattern = re.compile(regex)
        except re.error as e:
            print(f"Invalid regex: {e}", file=sys.stderr)
            return

        ns = self._get_caller_namespace()
        to_delete = [k for k in ns.keys() 
                     if not k.startswith('_') 
                     and k not in ['In', 'Out', '__builtins__'] 
                     and pattern.search(k)]
        
        if not to_delete:
            print(f"No variables matched regex '{regex}'.")
            return

        print(f"Deleting: {', '.join(to_delete)}")
        for var in to_delete:
            del ns[var]

    def _magic_alias_magic(self, args: str) -> None:
        """Create an alias for an existing magic."""
        parts = args.strip().split()
        if len(parts) != 2:
            print("Usage: %alias_magic <new_name> <target_name>")
            return
        
        new_name, target = parts
        new_name = new_name.lstrip('%')
        target = target.lstrip('%')
        
        # Verify target exists
        if not (hasattr(self, f"_magic_{target}") or hasattr(self, f"_cell_magic_{target}")):
            print(f"Error: Target magic '{target}' not found.", file=sys.stderr)
            return
            
        self.aliases[new_name] = target
        print(f"Created alias: %{new_name} -> %{target}")
    
    ## for matplotlib magics
    def _magic_matplotlib(self, args: str) -> None:
        """
        Set the matplotlib backend.
        Usage: %matplotlib [inline|widget|notebook]
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib is not installed.", file=sys.stderr)
            return

        arg = args.strip().lower()
        
        if not arg:
            print(f"Using matplotlib backend: {plt.get_backend()}")
            return

        if arg == 'inline':
            try:
                # In Pyodide, 'Agg' is often used to render to buffer
                plt.switch_backend('Agg') 
                print("Backend set to 'Agg' (inline equivalent)")
            except Exception as e:
                print(f"Error setting backend: {e}", file=sys.stderr)

        elif arg in ['widget', 'ipympl']:
            try:
                import ipympl
                plt.switch_backend('module://ipympl.backend_nbagg')
                print("Backend set to 'ipympl' (widget)")
            except ImportError:
                print("Error: 'ipympl' not installed.", file=sys.stderr)
            except Exception as e:
                print(f"Error setting backend: {e}", file=sys.stderr)
        
        else:
            print(f"Warning: Backend '{arg}' not fully supported. Trying anyway...", file=sys.stderr)
            try:
                plt.switch_backend(arg)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

    def _magic_reset(self, args: str) -> None:
        """Reset the namespace"""
        namespace = self._get_caller_namespace()
        to_delete = [k for k in namespace.keys() if not k.startswith('_') 
                     and k not in ['In', 'Out', '__builtins__', '__name__', '__doc__']]
        
        if not to_delete:
            print("Interactive namespace is already empty.")
            return

        if '-f' not in args:
            print(f"Once deleted, variables cannot be recovered. Proceed (y/[n])? ", end='')
            try:
                if input().lower() != 'y': return
            except: return
        
        for var in to_delete:
            try: del namespace[var]
            except: pass
        print(f"Deleted {len(to_delete)} variable(s).")

    # ============ CELL MAGICS ============
    
    def _cell_magic_time(self, code: str) -> None:
        """Time execution of entire cell"""
        if not code.strip(): return
        
        namespace = self._get_caller_namespace()
        start_time = time.perf_counter()
        start_process = time.process_time()
        
        try:
            exec(code, namespace)
        except Exception:
            traceback.print_exc()
        finally:
            wall_time = time.perf_counter() - start_time
            cpu_time = time.process_time() - start_process
            
            print(f"CPU times: total: {self._format_time(cpu_time)}")
            print(f"Wall time: {self._format_time(wall_time)}")
    
    def _cell_magic_timeit(self, code: str) -> None:
        """Time execution of entire cell with multiple runs."""
        if not code.strip(): return
        
        lines = code.split('\n')
        first_line = lines[0].strip()
        number = None
        repeat = 7
        code_start = 0
        
        if first_line.startswith('-'):
            number, repeat, _ = self._parse_timeit_args(first_line)
            code_start = 1
        
        code_to_time = '\n'.join(lines[code_start:])
        if not code_to_time.strip(): return
        
        namespace = self._get_caller_namespace()
        timer = timeit.Timer(code_to_time, globals=namespace)
        
        try:
            if number is None:
                for scale in [1, 10, 100, 1000]:
                    if timer.timeit(scale) >= 0.2:
                        number = scale
                        break
                else: number = 1000
            
            all_runs = timer.repeat(repeat=repeat, number=number)
            best = min(all_runs) / number
            stdev = statistics.stdev(all_runs) / number if len(all_runs) > 1 else 0
            
            self._print_timeit_result(best, stdev, repeat, number)
        except Exception as e:
            print(f"Error in timing: {e}", file=sys.stderr)
            traceback.print_exc()
            
    def _cell_magic_writefile(self, content: str) -> None:
        lines = content.split('\n', 1)
        filename = lines[0].strip()
        if not filename:
            print("Error: No filename specified", file=sys.stderr)
            return
        
        try:
            with open(filename, 'w') as f:
                f.write(lines[1] if len(lines) > 1 else "")
            print(f"Writing {filename}")
        except Exception as e:
            print(f"Error writing file: {e}", file=sys.stderr)

    def _cell_magic_capture(self, content: str) -> None:
        from contextlib import redirect_stdout, redirect_stderr
        namespace = self._get_caller_namespace()
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(content, namespace)
        except Exception:
            traceback.print_exc()
        
        namespace['_captured_stdout'] = stdout_capture.getvalue()
        namespace['_captured_stderr'] = stderr_capture.getvalue()
        print(f"Output captured in _captured_stdout and _captured_stderr")

    def _cell_magic_prun(self, code: str) -> None:
        """Run a cell through the python code profiler."""
        if not code.strip(): return
        self._run_with_profiler(code, self._get_caller_namespace())

# Create singleton instance
_handler = MagicHandler()

def line_magic(command: str, args: str) -> Any:
    return _handler.line_magic(command, args)

def cell_magic(command: str, args: str) -> Any:
    return _handler.cell_magic(command, args)