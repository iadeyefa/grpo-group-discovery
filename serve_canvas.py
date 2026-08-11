#!/usr/bin/env python3
"""
Root entrypoint relay for launching the canvas web server.
Delegates to scripts/serve_canvas.py
"""
import os
import sys

if __name__ == "__main__":
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', 'serve_canvas.py')
    with open(script_path) as f:
        code = compile(f.read(), script_path, 'exec')
        exec(code, {'__name__': '__main__', '__file__': script_path})
