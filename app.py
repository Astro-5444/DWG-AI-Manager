import os
import sys
import time
import json
import threading
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# =============================================================================
# CONFIGURATION
# =============================================================================

HAS_OCR_PIPELINE = True  # Optional OCR flag

# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_base_dir():
    """Get the directory where the EXE (or script) is located."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).parent
    return Path(__file__).parent.absolute()


def extract_project_name(source_path):
    """
    Extract project name from path.

    Example:
        'C:/.../AVIS/24101-MARRIOTT.../ICT/Drawings'
        -> '24101-MARRIOTT AUTOGRAPH PROJECT/09. ICT/Drawings'

    Args:
        source_path: Full path to the source folder

    Returns:
        Project name string or folder name if AVIS not found
    """
    try:
        path_obj = Path(source_path).absolute()
        parts = path_obj.parts

        if "AVIS" in parts:
            idx = parts.index("AVIS")
            if idx + 1 < len(parts):
                return "/".join(parts[idx + 1:])
    except Exception:
        pass

    return Path(source_path).name


def get_safe_filename(name):
    """Convert a path-like name into a safe filename by replacing slashes."""
    return name.replace("/", "_").replace("\\", "_")


# =============================================================================
# FILE TYPE UTILITIES
# =============================================================================

def is_dwg_file(filename):
    """Check if a file is a DWG drawing based on naming convention."""
    filename_upper = filename.upper()
    return filename_upper.endswith(".PDF") or filename_upper.endswith(".DWG")


def separate_dwg_files(source_folder):
    """
    Separate DWG files from regular files in a folder.

    Args:
        source_folder: Path to folder to scan

    Returns:
        Tuple of (dwg_files, regular_files) as lists of Path objects
    """
    dwg_files = []
    regular_files = []
    path = Path(source_folder)

    for item in path.iterdir():
        if not item.is_file():
            continue

        ext = item.suffix.lower()

        # PDF files - check if DWG
        if ext == ".pdf":
            if is_dwg_file(item.name):
                dwg_files.append(item)
            else:
                regular_files.append(item)

        # Office documents
        elif ext in {".docx", ".doc", ".xlsx", ".xls"}:
            regular_files.append(item)

    return dwg_files, regular_files


def count_files(folder):
    """Count all processable files in a folder."""
    supported_exts = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".dwg"}
    count = 0

    try:
        for item in Path(folder).rglob("*"):
            if item.is_file() and item.suffix.lower() in supported_exts:
                count += 1
    except Exception as e:
        print(f"Error counting files: {e}")

    return count


# =============================================================================
# FILE PROCESSING WORKERS
# =============================================================================

def process_single_dwg_wrapper(args):
    """
    Process a single DWG file with OCR (for parallel execution).

    Args:
        args: Tuple of (pdf_path, output_folder, image_root)

    Returns:
        Dictionary with success status, path, error, and filename
    """
    pdf_path, output_folder, image_root = args

    try:
        from OCR.pdf_ocr_pipeline import process_pdf

        pdf_path_obj = Path(pdf_path)
        # Write output file
        ocr_output_path = Path(output_folder) / f"{pdf_path_obj.stem}.txt"
        
        # SKIP if already exists
        if ocr_output_path.exists():
            log_message(f"⚡ OCR text already exists for {pdf_path_obj.name} – skipping.")
            return {
                'success': True,
                'path': str(ocr_output_path),
                'error': None,
                'filename': pdf_path_obj.name
            }

        results = process_pdf(pdf_path, image_output_root=image_root)

        # Combine all OCR text
        all_text = [text.strip() for text in results.values() if text.strip()]
        ocr_content = "\n\n".join(all_text)

        # Create header
        header = (
            f"FILE REFERENCE:\n{pdf_path_obj.name.removesuffix('.pdf')}\n\n"
            f"ORIGINAL PATH:\n{pdf_path}\n\n"
            "----------------------------------------\n\n"
            "CONTENT:\n"
        )

        # Write output file
        with open(ocr_output_path, 'w', encoding='utf-8') as f:
            f.write(header + ocr_content)

        return {
            'success': True,
            'path': str(ocr_output_path),
            'error': None,
            'filename': pdf_path_obj.name
        }

    except Exception as e:
        with open("worker_errors.log", "a") as f:
            f.write(f"Worker Error DWG ({Path(pdf_path).name}): {str(e)}\n")

        return {
            'success': False,
            'path': None,
            'error': str(e),
            'filename': Path(pdf_path).name
        }


def process_single_regular_file(args):
    """
    Process a single regular file (PDF, DOCX, XLSX) for parallel execution.

    Args:
        args: Tuple of (file_path, target_folder)

    Returns:
        Dictionary with success status, path, error, and filename
    """
    file_path, target_folder = args

    try:
        # Ensure target folder exists
        Path(target_folder).mkdir(parents=True, exist_ok=True)

        from Text_Process.extract_text_from_todo import (
            extract_pdf_text,
            extract_docx_text,
            extract_excel_text
        )

        file_path_obj = Path(file_path)
        ext = file_path_obj.suffix.lower()

        # Write output file
        txt_path = Path(target_folder) / f"{file_path_obj.stem}.txt"
        
        # SKIP if already exists
        if txt_path.exists():
            log_message(f"⚡ Text already extracted for {file_path_obj.name} – skipping.")
            return {
                'success': True,
                'path': str(txt_path),
                'error': None,
                'filename': file_path_obj.name
            }

        # Extract text based on file type
        if ext == ".pdf":
            content = extract_pdf_text(file_path_obj)
        elif ext in {".docx", ".doc"}:
            content = extract_docx_text(file_path_obj)
        elif ext in {".xlsx", ".xls"}:
            content = extract_excel_text(file_path_obj)
        else:
            content = "[Unsupported file type]"

        # Create header
        header = (
            f"FILE REFERENCE:\n{file_path_obj.stem}\n\n"
            f"ORIGINAL PATH:\n{file_path}\n\n"
            "----------------------------------------\n\n"
            "CONTENT:\n"
        )

        # Write output file
        with open(txt_path, "w", encoding="utf-8") as out_f:
            out_f.write(header + content)

        return {
            'success': True,
            'path': str(txt_path),
            'error': None,
            'filename': file_path_obj.name
        }

    except Exception as e:
        with open("worker_errors.log", "a") as f:
            f.write(f"Worker Error ({Path(file_path).name}): {str(e)}\n")

        return {
            'success': False,
            'path': None,
            'error': str(e),
            'filename': Path(file_path).name
        }


# =============================================================================
# APPLICATION STATE MANAGEMENT
# =============================================================================

app_state = {
    "source_folder":    None,
    "file_count":       0,
    "processing":       False,
    "progress":         0.0,
    "current_stage":    0,   # 0=idle, 1=todo, 2=extract, 3=ai, 4=excel, 5=symbols
    "files_done":       0,
    "files_total":      0,
    "current_file":     "",
    "stage_detail":     "",
    "logs":             [],
    "output_path":      None,
    "error":            None,
    "dwg_count":        0,
    "regular_count":    0,
    # --- new for two-mode workflow ---
    "mode":             None,       # "excel" | "count"
    "images_ready":     False,      # True once DWG→PNG step has been done
    "output_dir":       None,       # str path to output/<project_name>
    "project_name":     None,
    # --- count mode intermediate data ---
    "symbols_list":     [],         # list of {name, path} after images ready
    "files_list":       [],         # list of {name, folder} subfolders with floor_plan.png
    "count_ready":      False,      # True when user has confirmed selection for count
    "count_symbols":    [],         # selected symbol names
    "count_files":      [],         # selected file/folder names
    "count_output_path": None,      # path to the count excel
}


def reset_state():
    """Reset application state to initial values."""
    global app_state
    app_state = {
        "source_folder":    None,
        "file_count":       0,
        "processing":       False,
        "progress":         0.0,
        "current_stage":    0,
        "files_done":       0,
        "files_total":      0,
        "current_file":     "",
        "stage_detail":     "",
        "logs":             [],
        "output_path":      None,
        "error":            None,
        "dwg_count":        0,
        "regular_count":    0,
        "mode":             None,
        "images_ready":     False,
        "output_dir":       None,
        "project_name":     None,
        "symbols_list":     [],
        "files_list":       [],
        "count_ready":      False,
        "count_symbols":    [],
        "count_files":      [],
        "count_output_path": None,
    }


def log_message(msg):
    """Add a timestamped message to the log."""
    timestamp = time.strftime('%H:%M:%S')
    app_state["logs"].append(f"[{timestamp}] {msg}")
    print(msg)


def update_status(stage, progress=None, files_done=0, files_total=0):
    """
    Update the current processing status.

    Args:
        stage: Current stage number (0-5)
        progress: Progress value (0.0 to 1.0)
        files_done: Number of files completed
        files_total: Total number of files
    """
    app_state["current_stage"] = stage
    app_state["files_done"]    = files_done
    app_state["files_total"]   = files_total

    if progress is not None:
        app_state["progress"] = min(1.0, max(0.0, progress))


# =============================================================================
# SHARED: ENSURE IMAGES ARE READY  (PDF→PNG via OCR pipeline)
# =============================================================================

def ensure_images_ready():
    """
    Run the DWG → PNG image conversion step if it hasn't been done yet.
    Sets app_state["images_ready"] = True when complete.
    Returns (output_dir, project_name, dwg_files, todo_json_path, todo_list).
    """
    source_folder = app_state["source_folder"]
    project_name  = app_state.get("project_name") or extract_project_name(source_folder)
    app_state["project_name"] = project_name

    # ── Pre-process: Convert DWG to PDF if needed ───────────────────────────
    from DWG_Process.DWG_TO_PDF import plot_dwg_to_pdf
    try:
        dwgs = [p for p in Path(source_folder).rglob("*") if p.suffix.lower() == ".dwg"]
        if dwgs:
            log_message(f"Checking {len(dwgs)} DWG files for missing PDFs...")
            for dwg_path in dwgs:
                pdf_target = dwg_path.with_suffix(".pdf")
                if not pdf_target.exists():
                    log_message(f"Converting missing PDF for: {dwg_path.name}")
                    plot_dwg_to_pdf(str(dwg_path), output_folder=str(dwg_path.parent))
    except Exception as e:
        log_message(f"⚠️ Warning during DWG conversion: {e}")

    base_dir   = get_base_dir()
    output_dir = base_dir / "output" / project_name
    images_dir = output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "results").mkdir(parents=True, exist_ok=True)

    app_state["output_dir"] = str(output_dir)

    # ── If todo list already exists, just re-load it ──────────────
    todo_json_candidates = list(output_dir.glob("*to do list*.json"))
    if todo_json_candidates:
        todo_json_path = str(todo_json_candidates[0])
        with open(todo_json_path, "r", encoding="utf-8") as f:
            todo_list = json.load(f)
        dwg_files = [Path(item["path"]) for item in todo_list if is_dwg_file(Path(item["path"]).name)]
        log_message("⚡ Todo list already exists – re-loading.")
        
        # Check if images are likely already processed
        # If at least some subfolders contain images, count them as ready
        has_subfolders = any(d.is_dir() for d in output_dir.iterdir() if d.name.lower() != "symbols")
        if has_subfolders:
            app_state["images_ready"] = True
            log_message("⚡ Images appear to exist – skipping conversion stage.")
            return output_dir, project_name, dwg_files, todo_json_path, todo_list

    # ── STAGE 1: CREATE TODO LIST ─────────────────────────────────────────────
    update_status(1, 0.02)
    log_message(f"Stage 1: Processing project '{project_name}'...")

    from Text_Process.create_todo import build_todo_list
    todo_json_path = build_todo_list(source_folder, output_dir=str(output_dir))
    log_message(f"Todo list created at: {todo_json_path}")

    with open(todo_json_path, "r", encoding="utf-8") as f:
        todo_list = json.load(f)

    # ── Figure out DWG vs regular ─────────────────────────────────────────────
    dwg_files     = []
    regular_files = []
    for item in todo_list:
        fp = Path(item["path"])
        if is_dwg_file(fp.name):
            dwg_files.append(fp)
        else:
            regular_files.append(fp)

    app_state["dwg_count"]     = len(dwg_files)
    app_state["regular_count"] = len(regular_files)
    log_message(f"Found {len(regular_files)} regular files and {len(dwg_files)} DWG files")

    # ── STAGE 2-DWG: Run OCR / image generation on DWG files ─────────────────
    if dwg_files:
        cpu_count       = mp.cpu_count()
        max_workers_ocr = min(max(1, cpu_count // 2), 8)

        update_status(2, 0.10)
        log_message(f"Converting {len(dwg_files)} DWG files to images (OCR)…")

        args_list = [
            (str(fp), str(output_dir / fp.stem), str(images_dir))
            for fp in dwg_files
        ]

        processed_count = 0
        failed_count    = 0

        with ProcessPoolExecutor(max_workers=max_workers_ocr) as executor:
            future_to_file = {
                executor.submit(process_single_dwg_wrapper, args): args[0]
                for args in args_list
            }
            for future in as_completed(future_to_file):
                result = future.result()
                processed_count += 1
                progress = 0.10 + (processed_count / len(dwg_files)) * 0.30
                update_status(2, progress,
                              files_done=processed_count,
                              files_total=len(dwg_files))
                if result['success']:
                    log_message(f"✓ OCR [{processed_count}/{len(dwg_files)}] {result['filename']}")
                    app_state["current_file"]  = result['filename']
                    app_state["stage_detail"]  = f"{processed_count}/{len(dwg_files)} DWG files converted"
                else:
                    log_message(f"✗ OCR [{processed_count}/{len(dwg_files)}] {result['filename']}: {result['error']}")
                    failed_count += 1

        log_message(f"DWG conversion complete: {processed_count - failed_count} ok, {failed_count} failed")
    else:
        log_message("No DWG files found – skipping image conversion.")

    app_state["images_ready"]  = True
    app_state["current_file"]  = ""
    app_state["stage_detail"]  = ""
    return output_dir, project_name, dwg_files, todo_json_path, todo_list


# =============================================================================
# MODE 1: EXCEL PIPELINE
# =============================================================================

def run_excel_pipeline():
    """
    Full pipeline: todo → extract text → AI → Excel report.
    The DWG image step is shared with the count pipeline.
    """
    global app_state

    try:
        output_dir, project_name, dwg_files, todo_json_path, todo_list = ensure_images_ready()

        todo_base_name = Path(todo_json_path).stem.replace(" to do list", "")
        base_dir       = get_base_dir()
        results_dir    = base_dir / "results"

        # ── AI executor setup ─────────────────────────────────────────────────
        from Text_Process.API_client import APIClient
        from Text_Process.process_texts  import process_file_with_ai

        mistral         = APIClient()
        max_workers_ai  = max(1, min(mistral.key_count * 2, 12))
        ai_executor     = ThreadPoolExecutor(max_workers=max_workers_ai)
        ai_futures      = []

        cpu_count           = mp.cpu_count()
        max_workers_regular = min(max(1, cpu_count - 1), 8)

        # ── Separate regular files ────────────────────────────────────────────
        regular_files = [Path(item["path"]) for item in todo_list
                         if not is_dwg_file(Path(item["path"]).name)]
        total_files   = len(todo_list)

        # ── STAGE 2A: REGULAR FILES ───────────────────────────────────────────
        if regular_files:
            update_status(2, 0.40)
            log_message(f"Processing {len(regular_files)} regular files in parallel…")

            args_list = [
                (str(fp), str(output_dir / fp.stem))
                for fp in regular_files
            ]

            processed_count = 0
            failed_count    = 0

            with ProcessPoolExecutor(max_workers=max_workers_regular) as executor:
                future_to_file = {
                    executor.submit(process_single_regular_file, args): args[0]
                    for args in args_list
                }
                for future in as_completed(future_to_file):
                    result = future.result()
                    processed_count += 1
                    progress = 0.40 + (processed_count / len(regular_files)) * 0.15
                    update_status(2, progress,
                                  files_done=processed_count,
                                  files_total=total_files)
                    if result['success']:
                        log_message(f"✓ [{processed_count}/{len(regular_files)}] {result['filename']}")
                        ai_futures.append(
                            ai_executor.submit(process_file_with_ai, mistral, Path(result['path']))
                        )
                    else:
                        log_message(f"✗ [{processed_count}/{len(regular_files)}] {result['filename']}: {result['error']}")
                        failed_count += 1

            log_message(f"Regular files done: {processed_count - failed_count} ok, {failed_count} failed")

        # Queue DWG txt files for AI too
        for fp in dwg_files:
            txt = output_dir / fp.stem / f"{fp.stem}.txt"
            if txt.exists():
                ai_futures.append(
                    ai_executor.submit(process_file_with_ai, mistral, txt)
                )

        # Update todo list status
        for item in todo_list:
            item["status"] = "Text Extracted"
        with open(todo_json_path, "w", encoding="utf-8") as f:
            json.dump(todo_list, f, indent=2)

        log_message("All text extraction complete.")

        # ── STAGE 3: AI PROCESSING ────────────────────────────────────────────
        update_status(3, 0.55)
        log_message("Stage 3: AI processing…")

        summary_txt  = output_dir / f"{todo_base_name}_summary.txt"
        ai_results   = []
        total_ai     = len(ai_futures)
        completed_ai = 0
        ai_errors    = []

        if total_ai > 0:
            for future in as_completed(ai_futures):
                res = future.result()
                ai_results.append(res)
                completed_ai += 1
                prog = 0.55 + (completed_ai / total_ai) * 0.30
                update_status(3, prog, files_done=completed_ai, files_total=total_ai)
                if res['success']:
                    log_message(f"✓ AI [{completed_ai}/{total_ai}]: {res['file_name']}")
                else:
                    log_message(f"✗ AI [{completed_ai}/{total_ai}]: {res['file_name']} – {res['error']}")
                    ai_errors.append(f"AI error in {res['file_name']}: {res['error']}")
        else:
            log_message("No files to process with AI.")

        ai_results.sort(key=lambda x: x['file_name'])
        with open(summary_txt, 'w', encoding='utf-8') as out_f:
            for i, res in enumerate(ai_results):
                out_f.write(f"FILE: {res['file_name']}\n")
                out_f.write(res['reply'] if res['success'] else f"ERROR: {res['error']}\n")
                out_f.write("\n")
                if i < len(ai_results) - 1:
                    out_f.write("\n" + "-" * 80 + "\n\n")

        ai_executor.shutdown(wait=False)

        if ai_errors and not ai_results:
            raise Exception("AI processing failed completely: " + "; ".join(ai_errors[:3]))

        # ── STAGE 4: EXCEL REPORT ─────────────────────────────────────────────
        update_status(4, 0.88)
        app_state["stage_detail"] = "Building Excel report…"
        log_message("Stage 4: Generating Excel report…")

        final_excel = results_dir / f"{get_safe_filename(project_name)}.xlsx"
        from Text_Process.excel_maker import build_excel_from_txt
        build_excel_from_txt(str(summary_txt), str(final_excel))

        update_status(4, 1.0)
        log_message(f"✅ Excel report saved: {final_excel.name}")

        app_state.update({
            "output_path":  str(final_excel),
            "processing":   False,
            "current_file": "",
            "stage_detail": "Done!",
        })
        log_message("✅ All Excel pipeline tasks completed!")

    except Exception as e:
        log_message(f"❌ ERROR: {str(e)}")
        try:
            ai_executor.shutdown(wait=False)
        except Exception:
            pass
        app_state.update({
            "processing":    False,
            "error":         str(e),
            "current_stage": 0,
            "current_file":  "",
            "stage_detail":  "",
        })


# =============================================================================
# MODE 2: COUNT PIPELINE  – Step A: prepare images + discover symbols/files
# =============================================================================

def run_count_prepare():
    """
    Step A of count mode:
      1. Ensure images are ready (DWG → PNG via OCR)
      2. Discover available information_box.png sources for the user to choose from
      3. Discover existing symbols in symbols/ folder
      4. Discover floor-plan files
      5. Store everything in app_state so the UI can show selection
    """
    global app_state

    try:
        output_dir, project_name, dwg_files, todo_json_path, todo_list = ensure_images_ready()

        symbols_dir = output_dir / "symbols"
        symbols_dir.mkdir(exist_ok=True)

        # ── Already have symbols? Use them directly ───────────────────────────
        shared_symbols = sorted(list(symbols_dir.glob("*.png")))

        # ── Collect symbol-extraction sources – two paths per subfolder ─────
        # AUTO  path priority : Information_Box.png > floor_plan.png > any image
        # MANUAL path priority: High_Resolution.png > Information_Box.png > floor_plan.png > any image
        AUTO_PRIORITY   = ["information_box.png", "floor_plan.png"]
        MANUAL_PRIORITY = ["high_resolution.png", "information_box.png", "floor_plan.png"]
        valid_img_exts  = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

        def _pick(files_dict, priority, fallback_dict):
            """Return the first file in priority that exists, or any image as fallback."""
            for pname in priority:
                if pname in files_dict:
                    return files_dict[pname]
            # generic fallback
            for f in fallback_dict.values():
                if f.suffix.lower() in valid_img_exts:
                    return f
            return None

        symbol_sources = []
        for sub in sorted(output_dir.iterdir()):
            if not sub.is_dir() or sub.name.lower() == "symbols":
                continue
            files_in_sub = {f.name.lower(): f for f in sub.iterdir() if f.is_file()}
            auto_path   = _pick(files_in_sub, AUTO_PRIORITY,   files_in_sub)
            manual_path = _pick(files_in_sub, MANUAL_PRIORITY, files_in_sub)
            if auto_path:
                entry = {"name": sub.name, "path": str(auto_path)}
                if manual_path and str(manual_path) != str(auto_path):
                    entry["manual_path"] = str(manual_path)
                symbol_sources.append(entry)

        app_state["symbol_sources"] = symbol_sources

        if shared_symbols:
            log_message(f"✅ Found {len(shared_symbols)} existing symbol(s) in symbols/ folder.")
        elif symbol_sources:
            log_message(
                f"🔍 Found {len(symbol_sources)} source image(s) for symbol extraction. "
                "Please select one."
            )
        else:
            log_message("⚠️  No symbols and no source images found. Add symbol PNG files manually.")

        # ── Discover floor-plan files ──────────────────────────────────────────
        files_list = []
        valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
        for sub in sorted(output_dir.iterdir()):
            if sub.is_dir() and sub.name.lower() != "symbols":
                # Check for any image file in the subfolder
                has_image = any(f.is_file() and f.suffix.lower() in valid_exts for f in sub.iterdir())
                if has_image:
                    files_list.append({"name": sub.name, "folder": str(sub)})

        # ── Store in state ─────────────────────────────────────────────────────
        app_state["symbols_list"] = [
            {"name": s.stem, "path": str(s)}
            for s in shared_symbols
        ]
        app_state["files_list"]   = files_list

        update_status(5, 0.5)
        app_state.update({
            "processing":   False,
            "current_file": "",
            "stage_detail": "Select symbols and files to count.",
        })
        log_message(
            f"✅ Ready: {len(shared_symbols)} symbol(s), {len(files_list)} floor plan(s) found."
        )

    except Exception as e:
        log_message(f"❌ ERROR (count prepare): {str(e)}")
        app_state.update({
            "processing":    False,
            "error":         str(e),
            "current_stage": 0,
            "current_file":  "",
            "stage_detail":  "",
        })


# =============================================================================
# MODE 2: COUNT PIPELINE  – Step B: run the actual counting
# =============================================================================

def run_count_execute():
    """
    Step B of count mode:
      Count selected symbols across selected files and write a count Excel.
      Skips any pair (file × symbol) or output Excel that already exists.
    """
    global app_state

    selected_symbols = app_state.get("count_symbols", [])  # list of symbol names
    selected_files   = app_state.get("count_files",   [])  # list of folder names

    if not selected_symbols or not selected_files:
        log_message("❌ No symbols or files selected.")
        app_state["processing"] = False
        return

    try:
        import sys as _sys, re
        _legend_dir = str(get_base_dir() / "Legend_Counter")
        if _legend_dir not in _sys.path:
            _sys.path.insert(0, _legend_dir)
        from symbol_counter import count_symbol

        output_dir  = Path(app_state["output_dir"])
        symbols_dir = output_dir / "symbols"

        # Build a lookup: name → path for selected symbols
        sym_lookup = {s["name"]: Path(s["path"]) for s in app_state["symbols_list"]}
        # Build a lookup: folder name → folder path for selected files
        file_lookup = {f["name"]: Path(f["folder"]) for f in app_state["files_list"]}

        # ── Pre-scan: determine which files already have their Excel output ────
        files_to_process = []
        files_skipped    = []
        for file_name in selected_files:
            excel_output_path = output_dir / f"{file_name}_Legend_Count.xlsx"
            if excel_output_path.exists():
                log_message(f"⏭️  '{file_name}' – Excel already exists, loading previous counts.")
                files_skipped.append(file_name)
                
                # Attempt to load counts from existing Excel to keep the summary complete
                try:
                    import pandas as pd
                    df = pd.read_excel(excel_output_path)
                    # Expecting columns like 'Symbol Name' and 'Count'
                    rows[file_name] = {}
                    for _, r_row in df.iterrows():
                        s_name = str(r_row.get("Symbol Name", ""))
                        s_count = int(r_row.get("Count", 0))
                        if s_name:
                            rows[file_name][s_name] = s_count
                except Exception as ex:
                    log_message(f"⚠️ Could not load data from existing Excel: {ex}")
                    files_to_process.append(file_name) # Fallback to re-process
            else:
                files_to_process.append(file_name)

        # ── Gather data ────────────────────────────────────────────────────────
        rows = {}
        total_ops = len(selected_symbols) * len(files_to_process)
        done_ops  = 0

        update_status(5, 0.55)

        for file_name in files_to_process:
            folder = file_lookup.get(file_name)
            if not folder:
                log_message(f"⚠️  Folder not found for '{file_name}' – skipped.")
                continue

            # ── Find the best image to scan ───────────────────────────────────────
            floor_plan = None
            if folder.is_file():
                floor_plan = folder
            else:
                # Priority: floor_plan.png > High_Resolution.png > Information_Box.png > any image
                candidates = ["floor_plan.png", "High_Resolution.png", "Information_Box.png", "Low_Resolution.png"]
                found_files = {f.name.lower(): f for f in folder.iterdir() if f.is_file()}
                
                for c in candidates:
                    if c.lower() in found_files:
                        floor_plan = found_files[c.lower()]
                        break
                
                if not floor_plan:
                    # Fallback to the first available image
                    img_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
                    for f in folder.iterdir():
                        if f.is_file() and f.suffix.lower() in img_exts:
                            floor_plan = f
                            break

            if not floor_plan:
                log_message(f"⏭️  '{file_name}' – No scannable image found, skipping.")
                done_ops += len(selected_symbols)
                continue

            rows[file_name] = {}
            counted_png = folder / "counted_symbols.png"

            for sym_name in selected_symbols:
                # ── SKIP if already in rows (loaded from existing Excel) ─────────
                if file_name in rows and sym_name in rows[file_name]:
                    log_message(f"  ⏭️ {file_name} | {sym_name} – already in record, skipping.")
                    done_ops += 1
                    continue

                sym_path = sym_lookup.get(sym_name)
                if not sym_path or not sym_path.exists():
                    log_message(f"⚠️  Symbol '{sym_name}' not found – skipped.")
                    rows[file_name][sym_name] = -1
                    done_ops += 1
                    continue

                log_message(f"  🔢 {file_name} | {sym_name}…")
                app_state["current_file"] = file_name
                app_state["stage_detail"] = f"Counting {sym_name}…"

                try:
                    count = count_symbol(
                        symbol_path     = str(sym_path),
                        floor_plan_path = str(floor_plan),
                        output_path     = str(counted_png),
                        log_fn          = log_message,
                    )
                    rows[file_name][sym_name] = count
                    log_message(f"  ✅ {sym_name}: {count}")
                except Exception as ce:
                    log_message(f"  ❌ {sym_name}: {ce}")
                    rows[file_name][sym_name] = -1

                done_ops += 1
                progress = 0.55 + (done_ops / max(total_ops, 1)) * 0.40
                update_status(5, progress,
                              files_done=done_ops,
                              files_total=max(total_ops, 1))

        # ── Aggregate and Write ONE Master Excel ──────────────────────────────
        all_json_data = []
        for file_name in rows:
            for sym_name in selected_symbols:
                count = rows[file_name].get(sym_name, 0)
                all_json_data.append({
                    "File Reference": file_name,
                    "Symbol Name": sym_name,
                    "Count": count
                })

        if all_json_data:
            from legend_excel_maker import create_excel_from_data
            excel_name = f"{output_dir.name}_Legend_Summary.xlsx"
            excel_output_path = output_dir / excel_name
            
            create_excel_from_data(
                file_json_data=all_json_data,
                symbols_folder=symbols_dir,
                output_excel=excel_output_path
            )
            log_message(f"✅ Master Excel created: {excel_name}")
            app_state["count_output_path"] = str(excel_output_path)
            app_state["output_path"]       = str(excel_output_path)
        else:
            log_message("⚠️ No data to write to Excel.")

        update_status(5, 1.0)
        skipped_msg = f" ({len(files_skipped)} already existed)" if files_skipped else ""
        app_state.update({
            "processing":        False,
            "current_file":      "",
            "stage_detail":      "Done!",
        })
        log_message(f"✅ All symbol counting tasks complete!{skipped_msg}")

    except Exception as e:
        log_message(f"❌ ERROR (count execute): {str(e)}")
        app_state.update({
            "processing":    False,
            "error":         str(e),
            "current_stage": 0,
            "current_file":  "",
            "stage_detail":  "",
        })


# =============================================================================
# FLASK APP
# =============================================================================

app = Flask(__name__)


# =============================================================================
# SYMBOL EXTRACTION HELPERS
# =============================================================================

def _get_legend_dir():
    import sys as _sys
    _legend_dir = str(get_base_dir() / "Legend_Counter")
    if _legend_dir not in _sys.path:
        _sys.path.insert(0, _legend_dir)
    return _legend_dir


def _launch_symbol_maker_and_refresh(src_path: str, sym_dir: str):
    """
    Launch the Symbol Maker GUI and – once the window is closed – refresh
    app_state["symbols_list"] so the UI poll loop picks up the new symbols.
    """
    _get_legend_dir()   # ensure sys.path is set
    from symbol_maker import launch_viewer
    launch_viewer(src_path, sym_dir)          # blocks until window is closed
    # ── Refresh symbols list ──────────────────────────────────────────────
    sym_folder   = Path(sym_dir)
    shared       = sorted(sym_folder.glob("*.png"))
    app_state["symbols_list"] = [
        {"name": s.stem, "path": str(s)} for s in shared
    ]
    log_message(f"✅ Symbol Maker closed – {len(shared)} symbol(s) now available.")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/set-folder', methods=['POST'])
def set_folder():
    reset_state()
    data        = request.get_json()
    folder_path = data.get("folder_path", "").strip()

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    path = Path(folder_path)
    if not path.exists():
        return jsonify({"error": "Folder does not exist"}), 400
    if not path.is_dir():
        return jsonify({"error": "Path is not a directory"}), 400

    file_count = count_files(path)
    if file_count == 0:
        return jsonify({"error": "No supported files (.pdf, .docx, etc.) found in folder"}), 400

    app_state["source_folder"] = str(path)
    app_state["file_count"]    = file_count
    return jsonify({"success": True, "file_count": file_count})


# ── Mode 1: Start Excel pipeline ──────────────────────────────────────────────
@app.route('/start-excel', methods=['POST'])
def start_excel():
    if not app_state["source_folder"]:
        return jsonify({"error": "No folder selected"}), 400
    if app_state["processing"]:
        return jsonify({"error": "Already processing"}), 400

    app_state["processing"]   = True
    app_state["mode"]         = "excel"
    app_state["logs"]         = []
    app_state["output_path"]  = None
    app_state["error"]        = None
    app_state["current_stage"]= 0
    app_state["progress"]     = 0.0

    thread = threading.Thread(target=run_excel_pipeline, daemon=True)
    thread.start()
    return jsonify({"success": True})


# ── Mode 2a: Start count-prepare (images + discovery) ─────────────────────────
@app.route('/start-count-prepare', methods=['POST'])
def start_count_prepare():
    if not app_state["source_folder"]:
        return jsonify({"error": "No folder selected"}), 400
    if app_state["processing"]:
        return jsonify({"error": "Already processing"}), 400

    app_state["processing"]    = True
    app_state["mode"]          = "count"
    app_state["logs"]          = app_state.get("logs") or []
    app_state["output_path"]   = None
    app_state["error"]         = None
    app_state["current_stage"] = 0
    app_state["progress"]      = 0.0
    app_state["count_ready"]   = False

    thread = threading.Thread(target=run_count_prepare, daemon=True)
    thread.start()
    return jsonify({"success": True})


# ── Mode 2b: Submit selection and run counting ─────────────────────────────────
@app.route('/start-count-execute', methods=['POST'])
def start_count_execute():
    if not app_state["source_folder"]:
        return jsonify({"error": "No folder selected"}), 400
    if app_state["processing"]:
        return jsonify({"error": "Already processing"}), 400

    data = request.get_json()
    selected_symbols = data.get("symbols", [])
    selected_files   = data.get("files",   [])

    if not selected_symbols:
        return jsonify({"error": "No symbols selected"}), 400
    if not selected_files:
        return jsonify({"error": "No files selected"}), 400

    app_state["count_symbols"] = selected_symbols
    app_state["count_files"]   = selected_files
    app_state["processing"]    = True
    app_state["count_ready"]   = True
    app_state["error"]         = None
    app_state["output_path"]   = None
    app_state["current_stage"] = 5
    app_state["progress"]      = 0.50

    thread = threading.Thread(target=run_count_execute, daemon=True)
    thread.start()
    return jsonify({"success": True})


@app.route('/state')
def get_state():
    return jsonify(app_state)


@app.route('/symbol-img/<name>')
def symbol_img(name):
    """Serve a symbol PNG by its stem name so the UI can display it."""
    output_dir = app_state.get("output_dir")
    if not output_dir:
        return jsonify({"error": "No output dir"}), 404
    sym_path = Path(output_dir) / "symbols" / f"{name}.png"
    if not sym_path.exists():
        return jsonify({"error": "Not found"}), 404
    return send_file(str(sym_path), mimetype="image/png")


@app.route('/symbol-sources')
def get_symbol_sources():
    """Return list of available information_box.png files the user can extract from."""
    sources = app_state.get("symbol_sources", [])
    return jsonify({"sources": sources})


@app.route('/browse-file')
def browse_file():
    """
    Open a native file dialog (tkinter) and return the selected image path.
    Falls back gracefully when tkinter / a display is unavailable.
    """
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
                ("All files",   "*.*"),
            ]
        )
        root.destroy()

        if file_path:
            return jsonify({"file_path": file_path})
        return jsonify({"file_path": None})
    except Exception as e:
        return jsonify({"error": str(e), "file_path": None}), 500


@app.route('/extract-symbols', methods=['POST'])
def extract_symbols():
    """
    Extract symbols from a chosen information_box.png.
    Falls back to launching symbol_maker (PyQt5 GUI) if table detection fails.
    """
    data      = request.get_json()
    file_path = data.get("file_path", "").strip()
    is_manual = data.get("manual", False)

    if not file_path:
        return jsonify({"error": "No file_path provided"}), 400

    src = Path(file_path)
    if not src.exists():
        return jsonify({"error": f"File not found: {file_path}"}), 404

    output_dir = app_state.get("output_dir")
    if not output_dir:
        return jsonify({"error": "No output directory – run Count first"}), 400

    symbols_dir = Path(output_dir) / "symbols"
    symbols_dir.mkdir(parents=True, exist_ok=True)

    # ── Ensure the source's parent folder is in the scan list ─────────────
    # Add the parent directory (not the file itself) so the counting step
    # can find floor_plan.png / High_Resolution.png etc. inside it.
    src_folder = src.parent if src.is_file() else src
    already_in = any(
        str(src_folder) == f["folder"] for f in app_state.get("files_list", [])
    )
    if not already_in and src_folder != Path(output_dir) and src_folder.name.lower() != "symbols":
        app_state.setdefault("files_list", []).append(
            {"name": src_folder.name, "folder": str(src_folder)}
        )
        log_message(f"➕ Added folder '{src_folder.name}' to scan list.")

    # ── Clear old symbols so fresh extraction always runs ─────────────────
    for old in symbols_dir.glob("*.png"):
        try:
            old.unlink()
        except Exception:
            pass

    _get_legend_dir()  # ensure Legend_Counter is on sys.path

    if is_manual:
        log_message(f"✏️  Manual extraction requested for '{src.name}' – opening Symbol Maker…")
        try:
            t = threading.Thread(
                target=_launch_symbol_maker_and_refresh,
                args=(str(src), str(symbols_dir)),
                daemon=True
            )
            t.start()
            return jsonify({
                "success": True,
                "fallback": True,
                "message": "Symbol Maker opened for manual extraction.",
                "files_list": app_state["files_list"]
            })
        except Exception as gui_err:
            return jsonify({"error": f"GUI could not launch: {gui_err}"}), 500

    # ── Try automatic extraction ──────────────────────────────────────────────
    try:
        from symbol_extractor import detect_tables, extract_left_column_symbols, crop_legend_table, table_bounding_box
        import cv2

        img = cv2.imread(str(src))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {src}")

        sorted_tables, h_rects, v_rects = detect_tables(img)

        if len(sorted_tables) < 2:
            # --- Fallback: open symbol_maker GUI so user can crop manually ---
            log_message(f"⚠️  Table not detected in '{src.name}' – opening Symbol Maker for manual extraction…")
            try:
                t = threading.Thread(
                    target=_launch_symbol_maker_and_refresh,
                    args=(str(src), str(symbols_dir)),
                    daemon=True
                )
                t.start()
                return jsonify({
                    "success": True,
                    "fallback": True,
                    "message": "Table not found – Symbol Maker opened for manual extraction.",
                    "files_list": app_state["files_list"]
                })
            except Exception as gui_err:
                return jsonify({"error": f"Table detection failed and GUI could not launch: {gui_err}"}), 500

        # ── Auto-extract from the 2nd biggest table ───────────────────────────
        target  = sorted_tables[1]
        symbols = extract_left_column_symbols(target, h_rects, v_rects, img)
        
        if not symbols:
            # --- Fallback: table found but no symbols inside it ---
            log_message(f"⚠️  No symbols found in table in '{src.name}' – opening Symbol Maker…")
            try:
                t = threading.Thread(
                    target=_launch_symbol_maker_and_refresh,
                    args=(str(src), str(symbols_dir)),
                    daemon=True
                )
                t.start()
                return jsonify({
                    "success": True,
                    "fallback": True,
                    "message": "No symbols extracted – Symbol Maker opened for manual extraction.",
                    "files_list": app_state["files_list"]
                })
            except Exception as gui_err:
                return jsonify({"error": f"Extraction failed and GUI could not launch: {gui_err}"}), 500

        saved   = []

        import cv2 as _cv2
        for sym_img, row_idx in symbols:
            fp = symbols_dir / f"symbol_{row_idx + 1}.png"
            _cv2.imwrite(str(fp), sym_img)
            saved.append(str(fp))

        # Also save the full legend table crop
        legend_crop = crop_legend_table(target, h_rects, v_rects, img)
        legend_path = symbols_dir / "legend_table.png"
        _cv2.imwrite(str(legend_path), legend_crop)
        saved.append(str(legend_path))

        # Refresh symbols_list in state
        shared_symbols = sorted(list(symbols_dir.glob("*.png")))
        app_state["symbols_list"] = [
            {"name": s.stem, "path": str(s)}
            for s in shared_symbols
        ]
        log_message(f"✅ Extracted {len(symbols)} symbol(s) from '{src.name}'.")
        return jsonify({
            "success": True, 
            "fallback": False, 
            "count": len(symbols),
            "symbols": [{"name": s.stem, "path": str(s)} for s in shared_symbols],
            "files_list": app_state["files_list"]
        })

    except Exception as e:
        log_message(f"❌ Extraction error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/refresh-symbols')
def refresh_symbols():
    """
    Re-scan the symbols/ folder and update app_state.
    Called by the UI after Symbol Maker is closed to pull in newly saved symbols.
    """
    output_dir = app_state.get("output_dir")
    if not output_dir:
        return jsonify({"error": "No output directory – run Count first"}), 400
    symbols_dir = Path(output_dir) / "symbols"
    shared = sorted(symbols_dir.glob("*.png"))
    app_state["symbols_list"] = [
        {"name": s.stem, "path": str(s)} for s in shared
    ]
    log_message(f"🔄 Symbols refreshed – {len(shared)} symbol(s) found.")
    return jsonify({
        "success": True,
        "symbols": app_state["symbols_list"]
    })


@app.route('/download')
def download_excel():
    path = app_state.get("output_path") or app_state.get("count_output_path")
    if not path or not Path(path).exists():
        return jsonify({"error": "File not ready"}), 404
    return send_file(path, as_attachment=True)


@app.route('/open-excel')
def open_excel():
    path = app_state.get("output_path") or app_state.get("count_output_path")
    if not path or not Path(path).exists():
        return jsonify({"error": "File not found"}), 404
    try:
        os.startfile(path)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"Failed to open file: {str(e)}"}), 500


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


if __name__ == '__main__':
    mp.freeze_support()
    sys.setrecursionlimit(2000)

    base_dir = get_base_dir()
    (base_dir / "output").mkdir(exist_ok=True)
    (base_dir / "results").mkdir(exist_ok=True)

    app.template_folder = resource_path("templates")
    app.static_folder   = resource_path("static")

    print(f"Starting Server… (Templates: {app.template_folder})")
    print(f"Using {max(1, mp.cpu_count() - 1)} CPU cores for parallel processing")
    app.run(host='127.0.0.1', port=5009, debug=False)