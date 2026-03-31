import pytesseract
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import sys
import os

if hasattr(sys, "_MEIPASS"):
    base = sys._MEIPASS
else:
    base = os.path.abspath(".")

tesseract_cmd = os.path.join(base, "Tesseract", "tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

os.environ["TESSDATA_PREFIX"] = os.path.join(
    base, "Tesseract", "tessdata"
)

print("Tesseract path:", tesseract_cmd)
print("Exists:", os.path.exists(tesseract_cmd))



# webview_app.py
import webview
import threading
import time
import sys
import subprocess
import os
from pathlib import Path

import multiprocessing

class API:
    def selectFolder(self):
        """
        Try to open a folder selection dialog using multiple methods
        to ensure compatibility and robustness on Windows.
        """
        print("[PY] selectFolder() called from JS")
        
        # Method 1: Try tkinter (usually the most reliable on Windows)
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder_path = filedialog.askdirectory(title="Select Input Folder")
            root.destroy()
            
            if folder_path:
                print(f"[PY] Selected folder (via tkinter): {folder_path}")
                return folder_path
            else:
                print("[PY] Selection cancelled or failed via tkinter")
        except Exception as e:
            print(f"[PY] Tkinter dialog failed: {e}")

        # Method 2: Fallback to pywebview's native dialog
        try:
            print("[PY] Attempting pywebview native dialog fallback...")
            if webview.windows:
                result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
                print(f"[PY] Native dialog result: {result}")
                if result:
                    if isinstance(result, (list, tuple)):
                        return result[0]
                    return str(result)
        except Exception as e:
            print(f"[PY] Native dialog failed as well: {e}")

        return None

    def selectFile(self):
        """
        Open a file selection dialog for image files.
        Returns the selected file path string, or None.
        """
        print("[PY] selectFile() called from JS")
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            file_path = filedialog.askopenfilename(
                title="Select Legend Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                    ("All files", "*.*"),
                ]
            )
            root.destroy()

            if file_path:
                print(f"[PY] Selected file (via tkinter): {file_path}")
                return file_path
            else:
                print("[PY] File selection cancelled")
        except Exception as e:
            print(f"[PY] Tkinter file dialog failed: {e}")

        return None

def run_server():
    """Run the Flask server logic."""
    from app import app, reset_state, get_base_dir
    # Ensure directories exist where the exe is
    base_dir = get_base_dir()
    (base_dir / "output").mkdir(exist_ok=True)
    (base_dir / "results").mkdir(exist_ok=True)
    (base_dir / "output" / "images").mkdir(parents=True, exist_ok=True)
    print(f"Starting AI Project Reader Server at {base_dir}...")
    app.run(host='127.0.0.1', port=5000, debug=False)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def start_server_process():
    """Start the server as a subprocess, correctly handling PyInstaller environments."""
    print("Starting Flask server as a subprocess...")
    
    if getattr(sys, 'frozen', False):
        # In PyInstaller, sys.executable is the path to the .exe itself
        # We call the .exe with the --server flag
        return subprocess.Popen([sys.executable, "--server"], creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        # Development mode: run app.py with the current python interpreter
        return subprocess.Popen([sys.executable, "app.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)

if __name__ == '__main__':
    # CRITICAL for PyInstaller multi-process apps
    multiprocessing.freeze_support()
    
    # Check if we are being called as the server worker
    if "--server" in sys.argv:
        run_server()
        sys.exit(0)

    # Main Process (GUI)
    server_process = start_server_process()
    
    # 🕒 5-second delay to open the webview
    print("Waiting 5 seconds for server to initialize...")
    time.sleep(5)
    
    # Create the webview window
    print("Launching Desktop UI...")
    window = webview.create_window(
        "AI Project Reader",
        "http://127.0.0.1:5009",
        width=1200,
        height=800,
        js_api=API()
    )

    def on_closed():
        print("Window closed. Shutting down server and exiting...")
        try:
            server_process.terminate()
            server_process.wait(timeout=2)
        except:
            if server_process:
                server_process.kill()
        os._exit(0)

    window.events.closed += on_closed

    # Start the GUI loop
    webview.start(debug=False)
