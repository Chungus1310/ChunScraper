"""
My code runner.

This takes the Python script the AI wrote and runs it safely in its own little box.
"""

import subprocess
import os
from typing import Dict, Callable, Optional

from logging_config import get_logger

logger = get_logger(__name__)


def run_script_in_dir(path: str, log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, any]:
    # This runs the generated scraper.
    # First, it installs the requirements, then it runs the script.
    
    def log(message):
        logger.info(message)
        if log_callback:
            log_callback(message)

    log(f"Starting script execution in directory: {path}")
    
    try:
        # First, install all the libraries listed in requirements.txt.
        log("Installing dependencies from requirements.txt...")
        install_result = subprocess.run(
            ["pip", "install", "-r", "requirements.txt"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=120  # Give it 2 minutes to install everything.
        )
        
        logger.debug(f"Pip install exit code: {install_result.returncode}")
        logger.debug(f"Pip install stdout: {install_result.stdout}")
        logger.debug(f"Pip install stderr: {install_result.stderr}")
        
        # If pip failed, I can't continue.
        if install_result.returncode != 0:
            combined_output = (install_result.stderr or "") + "\n" + (install_result.stdout or "")
            error_message = f"Failed to install dependencies: {combined_output.strip()}"
            log(error_message)
            raise Exception(error_message)
        
        log("Dependencies installed successfully.")
        
        # Now, run the main scraper script.
        log("Executing scraper.py...")
        script_result = subprocess.run(
            ["python", "scraper.py"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=60  # Only give it a minute to run.
        )
        
        log(f"Script execution completed with exit code: {script_result.returncode}")
        
        if script_result.stdout:
            logger.debug(f"Script stdout preview: {script_result.stdout[:500]}...")
        if script_result.stderr:
            logger.debug(f"Script stderr: {script_result.stderr}")
        
        return {
            "stdout": script_result.stdout,
            "stderr": script_result.stderr,
            "exit_code": script_result.returncode
        }
        
    except subprocess.TimeoutExpired:
        log("Script execution timed out after 60 seconds")
        return {
            "stdout": "",
            "stderr": "Script execution timed out after 60 seconds",
            "exit_code": 1
        }
    except Exception as e:
        log(f"Error during script execution: {str(e)}")
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": 1
        }