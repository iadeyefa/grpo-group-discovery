#!/usr/bin/env python3
"""
Bundle GRPO Canvas into a Single Self-Contained Standalone HTML File.
Can be emailed directly or opened in any browser offline/online without a local server.
"""

import os

def bundle_canvas():
    canvas_dir = "canvas"
    
    with open(os.path.join(canvas_dir, "index.html"), "r") as f:
        html = f.read()

    with open(os.path.join(canvas_dir, "styles.css"), "r") as f:
        css = f.read()

    with open(os.path.join(canvas_dir, "data.js"), "r") as f:
        data_js = f.read()

    with open(os.path.join(canvas_dir, "world_map_data.js"), "r") as f:
        map_js = f.read()

    with open(os.path.join(canvas_dir, "app.js"), "r") as f:
        app_js = f.read()

    # Replace stylesheet link with inline CSS
    html = html.replace('<link rel="stylesheet" href="styles.css">', f'<style>\n{css}\n</style>')

    # Replace script src tags with inline JavaScript
    html = html.replace('<script src="data.js"></script>', f'<script>\n{data_js}\n</script>')
    html = html.replace('<script src="world_map_data.js"></script>', f'<script>\n{map_js}\n</script>')
    html = html.replace('<script src="app.js"></script>', f'<script>\n{app_js}\n</script>')

    output_path = os.path.join(canvas_dir, "standalone_dashboard.html")
    with open(output_path, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Generated single-file standalone dashboard: {output_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    bundle_canvas()
