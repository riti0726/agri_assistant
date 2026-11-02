"""
app.py
------
Main entrypoint running Dashboard.py
"""
import runpy
from pathlib import Path

dashboard_file = Path(__file__).parent / "Dashboard.py"
runpy.run_path(str(dashboard_file), run_name="__main__")
