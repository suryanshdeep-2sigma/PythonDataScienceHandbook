"""
Robust IPython Magic Commands Implementation for Pyodide

This module provides Python implementations of IPython magic commands.
Place at: /textbook/alps/ipython/magics.py

Features:
- Proper argument parsing using shlex and argparse
- Correct namespace access for timeit and exec
- Pythonic implementation using standard library callables
- Handles nested loops, complex expressions, and edge cases
"""

import os
import sys
import time
import timeit
import runpy
import traceback
import shlex
import re
from typing import Any, Dict, Optional, Tuple
import statistics
import pdb
import cgitb

class MagicHandler:
    """Handles execution of magic commands with proper namespace management"""
    
    def __init__(self):
        self.last_execution_time = None

        # Default exception mode
        self.xmode = 'context' 
        
        # Save original hook just in case
        self._original_excepthook = sys.excepthook
        
        # Install our Global Exception Interceptor
        sys.excepthook = self._custom_excepthook
        
    def _get_caller_namespace(self) -> Dict[str, Any]:
        """
        Get the namespace where the magic was called from.
        Goes up the stack to find the actual user namespace.
        """
        # Go up 2 frames: 
        # 0 = _get_caller_namespace
        # 1 = _magic_* method
        # 2 = line_magic/cell_magic
        # 3 = exec_and_print_last or user code
        # 4 = to reach the notebook code
        try:
            frame = sys._getframe(4)
        except ValueError:
            frame = sys._getframe(3)

        ns= frame.f_globals.copy()
        ns.update(frame.f_locals)
        return ns
        
    def line_magic(self, command: str, args: str) -> Any:
        """Execute a line magic command"""
        method_name = f"_magic_{command}"
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(args)
        else:
            print(f"UsageError: Line magic function `%{command}` not found.", file=sys.stderr)
            return None
    
    def cell_magic(self, command: str, args: str) -> Any:
        """Execute a cell magic command"""
        method_name = f"_cell_magic_{command}"
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(args)
        else:
            print(f"UsageError: Cell magic function `%%{command}` not found.", file=sys.stderr)
            return None
        
    def _custom_excepthook(self, etype, evalue, tb):
        """
        Global exception handler that respects %xmode logic.
        """
        # Always save the traceback so %debug can find it later!
        sys.last_type = etype
        sys.last_value = evalue
        sys.last_traceback = tb

        # Handle different modes
        if self.xmode == 'plain':
            # Just the error message, no stack trace 
            print(f"{etype.__name__}: {evalue}", file=sys.stderr)
            
        elif self.xmode == 'verbose':
            # Detailed stack trace with local variable values
            try:
                # cgitb formats detailed tracebacks in plain text
                hook = cgitb.Hook(format="text", file=sys.stderr)
                hook.handle((etype, evalue, tb))
            except Exception:
                # Fallback if cgitb fails
                traceback.print_exception(etype, evalue, tb, file=sys.stderr)
                
        else: # 'context' (Default)
            # Standard Python traceback
            traceback.print_exception(etype, evalue, tb, file=sys.stderr)
    
    # ============ LINE MAGICS ============
    
    def _magic_pwd(self, args: str) -> str:
        """Print working directory"""
        cwd = os.getcwd()
        print(cwd)
        return cwd
    
    def _magic_cd(self, args: str) -> None:
        """Change directory"""
        path = args.strip()
        if not path:
            # No args, go to home
            path = os.path.expanduser("~")
        
        try:
            os.chdir(path)
            print(os.getcwd())
        except FileNotFoundError:
            print(f"Error: Directory not found: {path}", file=sys.stderr)
        except PermissionError:
            print(f"Error: Permission denied: {path}", file=sys.stderr)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
    
    def _magic_ls(self, args: str) -> None:
        """List directory contents"""
        path = args.strip() if args.strip() else "."
        
        try:
            items = os.listdir(path)
            if items:
                for item in sorted(items):
                    print(item)
            else:
                # Empty directory
                pass
        except FileNotFoundError:
            print(f"Error: Directory not found: {path}", file=sys.stderr)
        except PermissionError:
            print(f"Error: Permission denied: {path}", file=sys.stderr)
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
            and not callable(v) or k in ['input', 'print']  # Keep some callables
        }
        
        # Type filtering
        args_stripped = args.strip().lower()
        if args_stripped:
            type_map = {
                'str': str,
                'int': int,
                'float': float,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
            }
            if args_stripped in type_map:
                target_type = type_map[args_stripped]
                user_vars = {k: v for k, v in user_vars.items() if isinstance(v, target_type)}
        
        if user_vars:
            for var_name in sorted(user_vars.keys()):
                print(var_name, end='\t')
            print()  # Newline at end
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
                    # Get string representation
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
            # Execute the statement in the caller's namespace
            exec(args, namespace)
        except Exception as e:
            traceback.print_exc()
            return
        finally:
            wall_time = time.perf_counter() - start_time
            cpu_time = time.process_time() - start_process
            
            # Format all times
            cpu_disp = self._format_time(cpu_time)
            wall_disp = self._format_time(wall_time)

            # Format similar to IPython
            print(f"CPU times: total: {cpu_disp}")
            print(f"Wall time: {wall_disp}")
            self.last_execution_time = wall_time
    
    def _parse_timeit_args(self, args: str) -> Tuple[int, int, str]:
        """
        Parse timeit arguments properly.
        
        Returns: (number, repeat, statement)
        """
        number = None  # Auto-determine
        repeat = 7
        
        # Use shlex for proper shell-like parsing
        try:
            tokens = shlex.split(args)
        except ValueError:
            # If shlex fails, fall back to simple split
            tokens = args.split()
        
        stmt_tokens = []
        i = 0
        
        while i < len(tokens):
            token = tokens[i]
            
            if token == '-n' and i + 1 < len(tokens):
                try:
                    number = int(tokens[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
            elif token == '-r' and i + 1 < len(tokens):
                try:
                    repeat = int(tokens[i + 1])
                    i += 2
                    continue
                except ValueError:
                    pass
            elif token.startswith('-n'):
                # Handle -n1000 format
                try:
                    number = int(token[2:])
                    i += 1
                    continue
                except ValueError:
                    pass
            elif token.startswith('-r'):
                # Handle -r10 format
                try:
                    repeat = int(token[2:])
                    i += 1
                    continue
                except ValueError:
                    pass
            
            # Not a flag, part of statement
            stmt_tokens.append(token)
            i += 1
        
        # Reconstruct statement from tokens
        stmt = ' '.join(stmt_tokens) if stmt_tokens else args
        
        # If no tokens were flags, use original args
        if not stmt.strip():
            stmt = args
        
        return number, repeat, stmt
    
    def _magic_timeit(self, args: str) -> None:
        """
        Time execution of a statement with multiple runs.
        
        Properly handles:
        - Nested loops: for i in range(100): ...
        - Complex expressions
        - Proper namespace access
        - Auto-scaling of iterations
        """
        if not args.strip():
            print("Error: No statement to time", file=sys.stderr)
            return
        
        number, repeat, stmt = self._parse_timeit_args(args)
        namespace = self._get_caller_namespace()
        
        # Create timer with proper namespace
        timer = timeit.Timer(stmt, globals=namespace)
        
        try:
            # Auto-scale if number not specified
            if number is None:
                # Determine number of loops automatically
                for scale in [1, 10, 100, 1000, 10000, 100000, 1000000]:
                    try:
                        time_taken = timer.timeit(scale)
                        if time_taken >= 0.2:
                            number = scale
                            break
                    except:
                        number = 1
                        break
                else:
                    number = 1000000
            
            # Run the timing
            all_runs = timer.repeat(repeat=repeat, number=number)
            
            # Calculate statistics
            best = min(all_runs) / number
            worst = max(all_runs) / number
            mean_time = statistics.mean(all_runs) / number
            
            if len(all_runs) > 1:
                stdev = statistics.stdev(all_runs) / number
            else:
                stdev = 0
            
            # Format output similar to IPython
            self._print_timeit_result(best, stdev, repeat, number)
            
        except Exception as e:
            print(f"Error in timing: {e}", file=sys.stderr)
            traceback.print_exc()
    
    def _print_timeit_result(self, best: float, stdev: float, repeat: int, number: int) -> None:
        """Format and print timeit results"""
        # Choose appropriate unit
        if best < 1e-6:
            # Nanoseconds
            time_val = best * 1e9
            stdev_val = stdev * 1e9
            unit = "ns"
        elif best < 1e-3:
            # Microseconds
            time_val = best * 1e6
            stdev_val = stdev * 1e6
            unit = "µs"
        elif best < 1:
            # Milliseconds
            time_val = best * 1e3
            stdev_val = stdev * 1e3
            unit = "ms"
        else:
            # Seconds
            time_val = best
            stdev_val = stdev
            unit = "s"
        
        # Print result
        loops_text = "loop" if number == 1 else "loops"
        runs_text = "run" if repeat == 1 else "runs"
        
        print(f"{time_val:.2f} {unit} ± {stdev_val:.2f} {unit} per loop "
              f"(mean ± std. dev. of {repeat} {runs_text}, {number} {loops_text} each)")
    
    # Add this helper method to your class
    def _format_time(self, timespan: float, precision: int = 2) -> str:
        """
        Formats a time duration with appropriate units (ns, µs, ms, s).
        """
        if timespan >= 1:
            return f"{timespan:.{precision}f} s"
        elif timespan >= 1e-3:
            return f"{timespan * 1e3:.{precision}f} ms"
        elif timespan >= 1e-6:
            return f"{timespan * 1e6:.{precision}f} µs"
        else:
            return f"{timespan * 1e9:.{precision}f} ns"

    def _magic_run(self, args: str) -> None:
        """Run a Python file in the current namespace"""
        filename = args.strip()
        
        if not filename:
            print("Error: No file specified", file=sys.stderr)
            return
        
        if not os.path.exists(filename):
            print(f"Error: File not found: {filename}", file=sys.stderr)
            return
        
        namespace = self._get_caller_namespace()
        
        try:
            # Run in the caller's namespace
            with open(filename, 'r') as f:
                code = f.read()
            exec(code, namespace)
        except Exception as e:
            print(f"Error running {filename}:", file=sys.stderr)
            traceback.print_exc()
    
    def _magic_reset(self, args: str) -> None:
        """Reset the namespace (clear variables)"""
        namespace = self._get_caller_namespace()
        
        # Get list of user-defined variables
        to_delete = [
            k for k in namespace.keys() 
            if not k.startswith('_') 
            and k not in ['In', 'Out', '__builtins__', '__name__', '__doc__']
        ]
        
        if not to_delete:
            print("Interactive namespace is already empty.")
            return
        
        # Check for -f flag (force)
        if '-f' not in args:
            print(f"Once deleted, variables cannot be recovered. Proceed (y/[n])? ", end='')
            try:
                response = input()
                if response.lower() != 'y':
                    print("Nothing done.")
                    return
            except:
                print("\nCancelled.")
                return
        
        # Delete variables
        deleted_count = 0
        for var in to_delete:
            try:
                del namespace[var]
                deleted_count += 1
            except:
                pass
        
        print(f"Deleted {deleted_count} variable(s).")
    
    def _magic_xmode(self, args: str) -> None:
        """Set exception mode (placeholder)"""
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
        """
        Activate the interactive debugger.
        This works because your worker handles blocking input()!
        """
        # Check if we have a traceback saved from a previous crash
        if not hasattr(sys, 'last_traceback') or sys.last_traceback is None:
            print("No traceback has been saved. Run code that crashes first.", file=sys.stderr)
            return

        print("Entering interactive debugger (pdb). Type 'q' to quit.")
        
        # Launch debugger on the last crash
        pdb.post_mortem(sys.last_traceback)   
    
    def _magic_pip(self, args: str) -> None:
        """Install packages using micropip"""
        try:
            import micropip
        except ImportError:
            print("Error: micropip not available", file=sys.stderr)
            return
        
        if 'install' in args:
            packages = args.replace('install', '').strip().split()
            if not packages:
                print("Error: No packages specified", file=sys.stderr)
                return
            
            for pkg in packages:
                print(f"Installing {pkg}...")
                try:
                    return micropip.install(pkg)
                    print(f"Successfully installed {pkg}")
                except Exception as e:
                    print(f"Error installing {pkg}: {e}", file=sys.stderr)
        elif 'list' in args:
            print("Listing installed packages...")
            # micropip doesn't have a great list function, but we can try
            try:
                import importlib.metadata
                dists = importlib.metadata.distributions()
                for dist in dists:
                    print(f"{dist.name} ({dist.version})")
            except:
                print("Unable to list packages")
        else:
            print(f"Usage: %pip install <package>")
    
    # ============ CELL MAGICS ============
    
    def _cell_magic_time(self, code: str) -> None:
        """Time execution of entire cell"""
        if not code.strip():
            return
        
        namespace = self._get_caller_namespace()
        
        start_time = time.perf_counter()
        start_process = time.process_time()
        
        try:
            exec(code, namespace)
        except Exception as e:
            traceback.print_exc()
        finally:
            wall_time = time.perf_counter() - start_time
            cpu_time = time.process_time() - start_process
            
            # Format all times
            cpu_disp = self._format_time(cpu_time)
            wall_disp = self._format_time(wall_time)

            print(f"CPU times: total: {cpu_disp}")
            print(f"Wall time: {wall_disp}")
    
    def _cell_magic_timeit(self, code: str) -> None:
        """
        Time execution of entire cell with multiple runs.
        
        First line can contain flags: -n <number> -r <repeat>
        Remaining lines are the code to time.
        """
        if not code.strip():
            print("Error: No code to time", file=sys.stderr)
            return
        
        lines = code.split('\n')
        first_line = lines[0].strip()
        
        # Check if first line contains flags
        number = None
        repeat = 7
        code_start = 0
        
        if first_line.startswith('-'):
            # Parse flags from first line
            number, repeat, _ = self._parse_timeit_args(first_line)
            code_start = 1
        
        # Get the actual code to time
        code_to_time = '\n'.join(lines[code_start:])
        
        if not code_to_time.strip():
            print("Error: No code to time", file=sys.stderr)
            return
        
        namespace = self._get_caller_namespace()
        timer = timeit.Timer(code_to_time, globals=namespace)
        
        try:
            # Auto-scale if needed
            if number is None:
                for scale in [1, 10, 100, 1000]:
                    try:
                        time_taken = timer.timeit(scale)
                        if time_taken >= 0.2:
                            number = scale
                            break
                    except:
                        number = 1
                        break
                else:
                    number = 1000
            
            # Run timing
            all_runs = timer.repeat(repeat=repeat, number=number)
            
            # Calculate statistics
            best = min(all_runs) / number
            if len(all_runs) > 1:
                stdev = statistics.stdev(all_runs) / number
            else:
                stdev = 0
            
            self._print_timeit_result(best, stdev, repeat, number)
            
        except Exception as e:
            print(f"Error in timing: {e}", file=sys.stderr)
            traceback.print_exc()
    
    def _cell_magic_writefile(self, content: str) -> None:
        """Write cell content to a file"""
        if not content.strip():
            print("Error: No content to write", file=sys.stderr)
            return
        
        lines = content.split('\n', 1)
        
        # First line should be the filename
        filename = lines[0].strip()
        if not filename:
            print("Error: No filename specified", file=sys.stderr)
            return
        
        # Rest is the file content
        file_content = lines[1] if len(lines) > 1 else ""
        
        try:
            with open(filename, 'w') as f:
                f.write(file_content)
            print(f"Writing {filename}")
        except Exception as e:
            print(f"Error writing file: {e}", file=sys.stderr)
    
    def _cell_magic_html(self, content: str) -> None:
        """Display HTML content (placeholder for Pyodide environment)"""
        print("HTML content:")
        print(content)
        # In a real implementation with Pyodide, you'd use:
        # from js import document
        # element = document.createElement('div')
        # element.innerHTML = content
        # display(element)
    
    def _cell_magic_capture(self, content: str) -> None:
        """Capture stdout/stderr from cell execution"""
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        namespace = self._get_caller_namespace()
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(content, namespace)
        except Exception as e:
            traceback.print_exc()
        
        # Store captured output in namespace
        namespace['_captured_stdout'] = stdout_capture.getvalue()
        namespace['_captured_stderr'] = stderr_capture.getvalue()
        
        print(f"Output captured in _captured_stdout and _captured_stderr")


# Create singleton instance
_handler = MagicHandler()

def line_magic(command: str, args: str) -> Any:
    """Execute a line magic command"""
    return _handler.line_magic(command, args)

def cell_magic(command: str, args: str) -> Any:
    """Execute a cell magic command"""
    return _handler.cell_magic(command, args)