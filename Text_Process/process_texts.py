import sys
import io
import os
from pathlib import Path

# Force UTF-8 output for Windows terminals
# Force UTF-8 output for Windows terminals (only if stdout exists)
if sys.stdout is not None and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from .API_client import APIClient

EXTRACTION_PROMPT = """From the provided text, extract **only** the following items **using the text itself exactly as written**:

### **Fields to Extract**

* **File PATH**
* **File Reference**
* **Project Name**
* **Drawing Title**
* **Floor** *(only if explicitly labeled or inferable from other fields)*
* **Block** *(only if explicitly labeled or inferable from other fields — may also appear as "Part")*
* **ISSUE DATE** *(only if explicitly labeled)*
* **REV.** *(only if explicitly labeled)*

---

### **Strict Rules**

1. **Do NOT infer or guess anything.**
2. **Do NOT extract from codes, numbers, filenames, or abbreviations** unless the item is **explicitly labeled** in the text
   (examples: `Floor:`, `Block:`, `Project Name:`).
3. If an item is **not explicitly mentioned**, output exactly:
   **`Not mentioned`**
4. **Use the exact wording, capitalization, and punctuation** from the text.
5. **Ignore filenames and file paths** unless the content explicitly labels them as one of the required fields.
6. If a required field value (especially Drawing Title) is split across multiple consecutive lines
   under the same label, JOIN those lines into a single line separated by a single space.
   Do NOT add, remove, or reword text — only concatenate.

   ### Example 
    If the text is:

    SHEET TITLE:
    DATA & IP TELEPHONE SYSTEM-FIRST FLOOR
    OVERALL PLAN

    Then extract:

    Drawing Title: DATA & IP TELEPHONE SYSTEM-FIRST FLOOR OVERALL PLAN

7. If a required field (Floor, Block, or Part) is **not explicitly labeled**,
   you MAY extract it ONLY if the exact wording appears elsewhere in the provided text
   (including Drawing Title or Sheet Title).
   The extracted value must match the text EXACTLY as written.
   Do NOT infer, summarize, normalize, or generate missing information.

   ### Example

   If the text contains:
   Drawing Title:
   DATA & IP TELEPHONE SYSTEM – FIRST FLOOR OVERALL PLAN

   And there is NO explicit "Floor:" label,
   Then output:
   Floor: FIRST FLOOR

   If the wording does NOT appear anywhere in the text,
   output:
   Floor: Not mentioned

8. If a required field (Floor, Block, or Part) cannot be found anywhere in the provided text,
   use the **File PATH or File Reference** as a last-resort source — but ONLY if the value
   appears explicitly and unambiguously as a recognizable word or phrase within the path/filename.
   Extract the value exactly as it appears in the path. Do NOT interpret abbreviations,
   codes, or short tokens unless they spell out the full word clearly.

   ### Example

   If Block is not found anywhere in the text body, but the File PATH is:
   D:\Projects\BLOCK C\Drawings\EL-101.dwg

   Then output:
   Block: BLOCK C

   If the path contains only an ambiguous code like "BLK-C" or "B3" with no clear word,
   output:
   Block: Not mentioned

---

### **Important Notes**

* **Block** may not be written as "Block"; it can be labeled as **"Part"**.
* **File PATH** may not be written as "File PATH"; it can be labeled as **"ORIGINAL PATH"**.
* **There is no such a thing "PART -A" it is "PART A"**.
* **Some information might be like this "234 | 12/08/2013 Issued For Construction AH
REV. DATE REASON FOR ISSUE CHK." So the issue date will be 12/08/2013 and REV will be 234.

---

### **Required Output Format**
```
File PATH: <value>
FILE REFERENCE: <value>
Project Name: <value>
Drawing Title: <value>
Floor: <value>
Block: <value or Not mentioned>
ISSUE DATE: <value>
REV.: <value>
```

---

Here is the text to process:

{content}"""

SYSTEM_PROMPT = """
You are a strict Engineer assistant.
You are an expert in civil engineering.
Respond in structured format.
No emojis. No fluff.
Extract only what is explicitly stated.
Do NOT add anything that is not in the provided text.
Do NOT infer, assume, or generate information.
Do NOT add explanations, notes, or commentary.
Output ONLY the requested format with extracted data.
If something is not explicitly mentioned, write "Not mentioned" - do not try to fill it in.
"""

import concurrent.futures

def process_file_with_ai(mistral, file_path):
    """Helper to process a single file with Mistral AI."""
    try:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Format prompt with file content
        user_text = EXTRACTION_PROMPT.format(content=content)
        
        # Get response from Mistral
        reply = mistral.chat(
            user_text=user_text,
            system_prompt=SYSTEM_PROMPT
        )
        return {"success": True, "file_name": file_path.name, "reply": reply}
    except Exception as e:
        return {"success": False, "file_name": file_path.name, "error": str(e)}

def process_folder(folder_path, output_file, progress_callback=None, max_workers=None):
    """
    Process all .txt files in a folder and extract structured information in parallel.
    
    Args:
        folder_path: Path to folder containing .txt files
        output_file: Path to output .txt file
        progress_callback: Optional progress update function
        max_workers: Number of concurrent API calls (defaults to 2x number of keys)
    
    Returns:
        dict with 'success' (bool), 'processed' (int), 'errors' (list)
    """
    mistral = APIClient()
    
    # Auto-calculate workers based on available keys
    if max_workers is None:
        # A safe ratio is 2 workers per key for Mistral's basic tiers
        max_workers = min(mistral.key_count * 2, 12)
        if max_workers < 1: max_workers = 1
    
    # Get all .txt files from folder
    folder = Path(folder_path)
    
    if not folder.exists():
        return {
            'success': False,
            'processed': 0,
            'errors': [f"Folder does not exist: {folder_path}"]
        }
    
    txt_files = sorted(folder.glob('*.txt'))
    
    if not txt_files:
        return {
            'success': False,
            'processed': 0,
            'errors': [f"No .txt files found in {folder_path}"]
        }
    
    errors = []
    processed_count = 0
    total_files = len(txt_files)
    
    print(f"Starting parallel processing with {max_workers} workers...", flush=True)

    results = [None] * total_files
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map of future to index to keep order
            future_to_idx = {executor.submit(process_file_with_ai, mistral, txt_files[i]): i 
                            for i in range(total_files)}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                result = future.result()
                results[idx] = result
                
                # Progress update
                completed = sum(1 for r in results if r is not None)
                if progress_callback:
                    progress_callback(completed, total_files, result['file_name'])
                
                if result['success']:
                    print(f"  ✓ [{completed}/{total_files}] {result['file_name']} completed", flush=True)
                else:
                    print(f"  ✗ [{completed}/{total_files}] {result['file_name']} failed: {result['error']}", flush=True)

        # Write results in order to the output file
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for i, result in enumerate(results):
                if result is None: continue # Should not happen
                
                out_f.write(f"FILE: {result['file_name']}\n")
                if result['success']:
                    out_f.write(result['reply'])
                    processed_count += 1
                else:
                    out_f.write(f"ERROR: {result['error']}\n")
                    errors.append(f"Error processing {result['file_name']}: {result['error']}")
                
                out_f.write("\n")
                if i < total_files - 1:
                    out_f.write("\n" + "-" * 80 + "\n\n")

        print(f"\n✓ Done! Processed {processed_count}/{total_files} files", flush=True)
        print(f"✓ Output saved to: {output_file}", flush=True)
        
        return {
            'success': True,
            'processed': processed_count,
            'errors': errors
        }
        
    except Exception as e:
        error_msg = f"Fatal error during parallel processing: {str(e)}"
        print(f"\n✗ {error_msg}", flush=True)
        return {
            'success': False,
            'processed': processed_count,
            'errors': [error_msg]
        }


if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "input_folder"  # Change this
    OUTPUT_FILE = "extracted_data.txt"  # Change this
    
    # Process the folder
    result = process_folder(FOLDER_PATH, OUTPUT_FILE)
    
    # Exit with error code if failed
    if not result['success'] or result['errors']:
        sys.exit(1)
