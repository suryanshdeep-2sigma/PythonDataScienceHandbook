"""
Example implementation of alps.ipython.magics module

This should be placed in your Python package structure as:
/textbook/alps/ipython/magics.py (or similar)

This module provides the Python implementations that magic commands
will be transformed into.
"""

import os
import sys
import time
import timeit
import runpy
import traceback
from typing import Any, Dict
import json


class MagicHandler:
    """Handles execution of magic commands"""
    
    def __init__(self):
        self.last_execution_time = None
        
    def line_magic(self, command: str, args: str) -> Any:
        """
        Execute a line magic command
        
        Args:
            command: The magic command name (without %)
            args: The arguments as a string
            
        Returns:
            Result of the magic command (if any)
        """
        method_name = f"_magic_{command}"
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(args)
        else:
            raise ValueError(f"Unknown line magic: %{command}")
    
    def cell_magic(self, command: str, args: str) -> Any:
        """
        Execute a cell magic command
        
        Args:
            command: The magic command name (without %%)
            args: The cell content as a string
            
        Returns:
            Result of the magic command (if any)
        """
        method_name = f"_cell_magic_{command}"
        
        if hasattr(self, method_name):
            return getattr(self, method_name)(args)
        else:
            raise ValueError(f"Unknown cell magic: %%{command}")
    
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
        os.chdir(path)
        print(os.getcwd())
    
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
        frame = sys._getframe(1)
        user_vars = {k: v for k, v in frame.f_globals.items() 
                     if not k.startswith('_') and k not in ['In', 'Out']}
        
        if args.strip() == 'str':
            # Only string variables
            user_vars = {k: v for k, v in user_vars.items() if isinstance(v, str)}
        elif args.strip() == 'int':
            user_vars = {k: v for k, v in user_vars.items() if isinstance(v, int)}
        
        if user_vars:
            for var_name in sorted(user_vars.keys()):
                print(var_name)
        else:
            print("Interactive namespace is empty.")
    
    def _magic_whos(self, args: str) -> None:
        """List variables with more detail"""
        frame = sys._getframe(1)
        user_vars = {k: v for k, v in frame.f_globals.items() 
                     if not k.startswith('_') and k not in ['In', 'Out']}
        
        if user_vars:
            print(f"{'Variable':<20} {'Type':<20} {'Data/Info':<30}")
            print("-" * 70)
            for var_name in sorted(user_vars.keys()):
                var = user_vars[var_name]
                var_type = type(var).__name__
                try:
                    data_info = str(var)[:30]
                except:
                    data_info = "<unprintable>"
                print(f"{var_name:<20} {var_type:<20} {data_info:<30}")
        else:
            print("Interactive namespace is empty.")
    
    def _magic_time(self, args: str) -> None:
        """Time execution of a statement"""
        frame = sys._getframe(1)
        start = time.perf_counter()
        
        try:
            exec(args, frame.f_globals, frame.f_locals)
        finally:
            elapsed = time.perf_counter() - start
            print(f"CPU times: user {elapsed:.2f} s, sys: 0.00 s, total: {elapsed:.2f} s")
            print(f"Wall time: {elapsed:.2f} s")
            self.last_execution_time = elapsed
    
    def _magic_timeit(self, args: str) -> None:
        """Time execution of a statement multiple times"""
        # Parse timeit arguments
        # Simple parsing - in production you'd want proper arg parsing
        parts = args.split()
        
        number = 100000
        repeat = 7
        stmt = args
        
        # Look for -n and -r flags
        i = 0
        while i < len(parts):
            if parts[i] == '-n' and i + 1 < len(parts):
                number = int(parts[i + 1])
                stmt = ' '.join(parts[:i] + parts[i+2:])
                i += 2
            elif parts[i] == '-r' and i + 1 < len(parts):
                repeat = int(parts[i + 1])
                stmt = ' '.join(parts[:i] + parts[i+2:])
                i += 2
            else:
                i += 1
        
        frame = sys._getframe(1)
        
        timer = timeit.Timer(stmt, globals=frame.f_globals)
        results = timer.repeat(repeat=repeat, number=number)
        best = min(results) / number
        
        # Format output similar to IPython
        if best < 1e-6:
            print(f"{best * 1e9:.2f} ns ± {max(results) * 1e9 - min(results) * 1e9:.2f} ns per loop (mean ± std. dev. of {repeat} runs, {number} loops each)")
        elif best < 1e-3:
            print(f"{best * 1e6:.2f} µs ± {(max(results) - min(results)) * 1e6:.2f} µs per loop (mean ± std. dev. of {repeat} runs, {number} loops each)")
        elif best < 1:
            print(f"{best * 1e3:.2f} ms ± {(max(results) - min(results)) * 1e3:.2f} ms per loop (mean ± std. dev. of {repeat} runs, {number} loops each)")
        else:
            print(f"{best:.2f} s ± {max(results) - min(results):.2f} s per loop (mean ± std. dev. of {repeat} runs, {number} loops each)")
    
    def _magic_run(self, args: str) -> None:
        """Run a Python file"""
        filename = args.strip()
        try:
            runpy.run_path(filename, run_name="__main__")
        except Exception as e:
            traceback.print_exc()
    
    def _magic_reset(self, args: str) -> None:
        """Reset namespace"""
        frame = sys._getframe(1)
        # Get list of user-defined variables
        to_delete = [k for k in frame.f_globals.keys() 
                     if not k.startswith('_') and k not in ['In', 'Out']]
        
        if '-f' not in args:
            response = input(f"Once deleted, variables cannot be recovered. Proceed (y/[n])? ")
            if response.lower() != 'y':
                print("Nothing done.")
                return
        
        for var in to_delete:
            try:
                del frame.f_globals[var]
            except:
                pass
        print("Variables cleared.")
    
    def _magic_xmode(self, args: str) -> None:
        """Set exception mode"""
        mode = args.strip().lower()
        if mode in ['plain', 'context', 'verbose']:
            # Store the mode preference
            # This would need integration with your error handler
            print(f"Exception reporting mode: {mode}")
            # You'd set a global or modify sys.excepthook here
        else:
            print("Valid modes: plain, context, verbose")
    
    def _magic_debug(self, args: str) -> None:
        """Enter debugger"""
        # This would need proper pdb integration
        print("Debug mode would be activated here")
        print("Note: pdb may not work fully in browser environment")
    
    def _magic_pip(self, args: str) -> None:
        """Run pip command"""
        import subprocess
        try:
            # In Pyodide, use micropip instead
            if 'install' in args:
                import micropip
                packages = args.replace('install', '').strip().split()
                for pkg in packages:
                    print(f"Installing {pkg}...")
                    micropip.install(pkg)
            else:
                print(f"Executing: pip {args}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
    
    # ============ CELL MAGICS ============
    
    def _cell_magic_time(self, code: str) -> None:
        """Time execution of entire cell"""
        frame = sys._getframe(1)
        start = time.perf_counter()
        
        try:
            exec(code, frame.f_globals, frame.f_locals)
        finally:
            elapsed = time.perf_counter() - start
            print(f"CPU times: user {elapsed:.2f} s, sys: 0.00 s, total: {elapsed:.2f} s")
            print(f"Wall time: {elapsed:.2f} s")
    
    def _cell_magic_timeit(self, code: str) -> None:
        """Time execution of entire cell multiple times"""
        # Extract options from first line if present
        lines = code.split('\n')
        first_line = lines[0].strip()
        
        number = 100
        repeat = 7
        
        # Simple option parsing
        if first_line.startswith('-'):
            options = first_line.split()
            code_lines = lines[1:]
            for i, opt in enumerate(options):
                if opt == '-n' and i + 1 < len(options):
                    number = int(options[i + 1])
                elif opt == '-r' and i + 1 < len(options):
                    repeat = int(options[i + 1])
            code = '\n'.join(code_lines)
        
        frame = sys._getframe(1)
        timer = timeit.Timer(code, globals=frame.f_globals)
        results = timer.repeat(repeat=repeat, number=number)
        best = min(results) / number
        
        if best < 1e-3:
            print(f"{best * 1e6:.2f} µs ± {(max(results) - min(results)) * 1e6:.2f} µs per loop")
        elif best < 1:
            print(f"{best * 1e3:.2f} ms ± {(max(results) - min(results)) * 1e3:.2f} ms per loop")
        else:
            print(f"{best:.2f} s ± {max(results) - min(results):.2f} s per loop")
    
    def _cell_magic_writefile(self, content: str) -> None:
        """Write cell content to file"""
        lines = content.split('\n')
        if not lines:
            print("Error: No filename specified", file=sys.stderr)
            return
        
        # First line (or part of it) is the filename
        first_line = lines[0].strip()
        if ' ' in first_line:
            # Filename is first word
            filename = first_line.split()[0]
            file_content = '\n'.join([first_line.split(' ', 1)[1]] + lines[1:])
        else:
            filename = first_line
            file_content = '\n'.join(lines[1:])
        
        try:
            with open(filename, 'w') as f:
                f.write(file_content)
            print(f"Writing {filename}")
        except Exception as e:
            print(f"Error writing file: {e}", file=sys.stderr)
    
    def _cell_magic_script(self, content: str) -> None:
        """Run cell content as a script"""
        lines = content.split('\n')
        if not lines:
            return
        
        # First word is the interpreter
        interpreter = lines[0].strip().split()[0]
        code = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        
        print(f"Running script with {interpreter}...")
        # In a real implementation, you'd execute this with the specified interpreter


# Create singleton instance
_handler = MagicHandler()

def line_magic(command: str, args: str) -> Any:
    """Execute a line magic"""
    return _handler.line_magic(command, args)

def cell_magic(command: str, args: str) -> Any:
    """Execute a cell magic"""
    return _handler.cell_magic(command, args)