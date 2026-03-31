from pathlib import Path
import json


def build_todo_list(folder_path, output_dir=None):
    exts = {".pdf", ".docx", ".doc", ".xlsx", ".xls"}

    folder = Path(folder_path).resolve()

    if not folder.exists() or not folder.is_dir():
        raise ValueError("Input must be an existing folder")

    # Folder names (closest first)
    folder1 = folder.name
    folder2 = folder.parent.name
    folder3 = folder.parent.parent.name

    output_filename = f"{folder3}-{folder2}-{folder1}-to do list.json"
    
    if output_dir:
        output_path = Path(output_dir) / output_filename
    else:
        output_path = Path(output_filename)

    todo_items = []

    for item in folder.rglob("*"):
        if item.is_file() and item.suffix.lower() in exts:
            todo_items.append({
                "path": str(item),
                "filename": item.name,
                "status": "pending"
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(todo_items, f, indent=2)

    return str(output_path)
