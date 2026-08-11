#!/usr/bin/env python3
"""
HTTP Server script to serve the GRPO Group Discovery Deep Evaluation Canvas.
Can be executed directly or via python3 scripts/serve_canvas.py
"""
import os
import sys
import http.server
import socketserver

PORT = 8050

def get_canvas_dir():
    # Check current directory
    if os.path.exists('canvas'):
        return os.path.abspath('canvas')
    # Check parent directory if script is inside scripts/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_canvas = os.path.join(os.path.dirname(script_dir), 'canvas')
    if os.path.exists(parent_canvas):
        return parent_canvas
    raise FileNotFoundError("Could not locate 'canvas' directory.")

class CanvasHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        canvas_dir = get_canvas_dir()
        super().__init__(*args, directory=canvas_dir, **kwargs)

def main():
    canvas_dir = get_canvas_dir()
    os.chdir(canvas_dir)
    with socketserver.TCPServer(("", PORT), CanvasHandler) as httpd:
        print(f"🚀 GRPO Group Discovery Canvas running at: http://localhost:{PORT}")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()
