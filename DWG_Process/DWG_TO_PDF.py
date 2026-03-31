import os
import subprocess

def get_active_layout_name(dwg_path, acad_console=r"C:\Program Files\Autodesk\AutoCAD 2026\accoreconsole.exe"):
    """
    Finds the active layout (CTAB) of a DWG file using AutoCAD Core Console.
    Helper for correctly labeling/targeting plots.
    """
    if not os.path.exists(dwg_path):
        return None

    dwg_folder = os.path.dirname(dwg_path)
    output_file = os.path.join(dwg_folder, "active_layout.txt")
    scr_path = os.path.join(dwg_folder, "get_active.scr")

    scr_content = f"""(setq f (open "{output_file.replace("\\", "/")}" "w"))
(if f (progn (write-line (getvar "CTAB") f) (close f)))
_.QUIT
_Y
"""
    with open(scr_path, "w") as f:
        f.write(scr_content)

    active_layout = None
    try:
        result = subprocess.run(
            [acad_console, "/i", dwg_path, "/s", scr_path],
            capture_output=True, text=True  # No check=True — we handle it ourselves
        )
        if result.returncode != 0:
            print(f"  ⚠ accoreconsole returned code {result.returncode} during layout detection")
            print("  --- STDOUT ---"); print(result.stdout)
            print("  --- STDERR ---"); print(result.stderr)

        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                active_layout = f.read().strip()
    except Exception as e:
        print(f"  ⚠ Could not detect active layout: {e}")
    finally:
        if os.path.exists(scr_path):  os.remove(scr_path)
        if os.path.exists(output_file): os.remove(output_file)

    return active_layout


def plot_dwg_to_pdf(dwg_path, output_folder=None,
                     acad_console=r"C:\Program Files\Autodesk\AutoCAD 2026\accoreconsole.exe",
                     pc3="AutoCAD PDF (High Quality Print).pc3",
                     paper_size="ISO full bleed B0 (1414.00 x 1000.00 MM)",
                     layout_name=None):

    if not os.path.isfile(dwg_path):
        raise FileNotFoundError(f"DWG file not found: {dwg_path}")

    # Detect active layout if not provided
    if layout_name is None:
        print(f"  🔍 Detecting active layout for {os.path.basename(dwg_path)}...")
        layout_name = get_active_layout_name(dwg_path, acad_console)
        if layout_name:
            print(f"  💡 Active layout found: '{layout_name}'")
        else:
            print(f"  ⚠ Could not detect layout — AutoCAD will use its default.")
            layout_name = ""

    dwg_folder = os.path.dirname(dwg_path)
    if output_folder is None:
        output_folder = os.path.join(dwg_folder, "Output", "PDF_From_DWG")
    os.makedirs(output_folder, exist_ok=True)

    pdf_name = os.path.splitext(os.path.basename(dwg_path))[0] + ".pdf"
    pdf_path = os.path.join(output_folder, pdf_name)

    if layout_name.upper() == "MODEL":
        plot_cmds = [
            "-PLOT", "Y", "Model", pc3, paper_size, "M", "L", "N", "E", "F", "C",
            "Y", "acad.ctb", "Y", "N",
            f'"{pdf_path.replace("\\", "/")}"',
            "N", "Y"
        ]
    else:
        # Sequence matched to the EXACT prompts AutoCAD 2026 issues for a paper space layout:
        # 1.  Detailed plot configuration? → Y
        # 2.  Layout name                  → layout_name
        # 3.  Output device               → pc3
        # 4.  Paper size                  → paper_size
        # 5.  Paper units                 → M  (Millimeters)
        # 6.  Drawing orientation         → L  (Landscape)
        # 7.  Plot upside down?           → N
        # 8.  Plot area                   → L  (Layout)
        # 9.  Plot scale                  → F  (Fit)
        # 10. Plot offset                 → 0,0  (using explicit coords — "C" caused AutoCAD to
        #                                         misread the next answer as a new command)
        # 11. Plot with plot styles?      → Y
        # 12. Plot style table name       → monochrome.ctb
        # 13. Plot with lineweights?      → Y  (applies lineweights from CTB)
        # 14. Scale lineweights?          → N
        # 15. Plot paper space first?     → N
        # 16. Hide paperspace objects?    → N
        # 17. Write to file? / filename   → pdf_path
        # 18. Save changes to layout?     → N
        # 19. Proceed with plot?          → Y
        plot_cmds = [
            "-PLOT",
            "Y",            # Detailed plot configuration?
            layout_name,    # Layout name
            pc3,            # Output device
            paper_size,     # Paper size
            "M",            # Millimeters
            "L",            # Landscape
            "N",            # Plot upside down? No
            "L",            # Plot area: Layout
            "F",            # Scale: Fit
            "0,0",          # Plot offset: explicit 0,0 instead of "C" to avoid prompt ambiguity
            "Y",            # Plot with plot styles?
            "monochrome.ctb",
            "Y",            # Plot with lineweights?
            "N",            # Scale lineweights?
            "N",            # Plot paper space first?
            "N",            # Hide paperspace objects?
            f'"{pdf_path.replace("\\", "/")}"',  # Output filename
            "N",            # Save changes to layout?
            "Y"             # Proceed with plot?
        ]

    scr_content = "\n".join(plot_cmds) + "\nQUIT\nY\n"
    scr_path = os.path.join(dwg_folder, "temp_plot.scr")

    with open(scr_path, "w") as f:
        f.write(scr_content)

    # ── Print exactly what we're sending ──────────────────────────────────────
    print(f"\n  🖥  Command : {acad_console} /i {dwg_path} /s {scr_path}")
    print(f"  📄 Script  :\n{'─'*60}")
    for i, line in enumerate(scr_content.splitlines(), 1):
        print(f"    {i:>2}: {line}")
    print(f"{'─'*60}\n")

    try:
        result = subprocess.run(
            [acad_console, "/i", dwg_path, "/s", scr_path],
            capture_output=True, text=True
            # ⚠ No check=True — we ALWAYS want to see the output, success or failure
        )

        # ── Always print the full AutoCAD dialogue ────────────────────────────
        print(f"{'─'*25} AutoCAD STDOUT {'─'*25}")
        print(result.stdout if result.stdout.strip() else "  (empty)")
        print(f"{'─'*25} AutoCAD STDERR {'─'*25}")
        print(result.stderr if result.stderr.strip() else "  (empty)")
        print(f"{'─'*65}")
        print(f"  Return code: {result.returncode}")

        # ── Check whether the PDF was actually produced ───────────────────────
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            print(f"  ✅ PDF created : {pdf_path}")
            return pdf_path
        else:
            print(f"  ❌ PDF was NOT created at: {pdf_path}")
            print("     Check the AutoCAD dialogue above — a prompt was likely unanswered.")
            return None

    except Exception as e:
        print(f"  ❌ Unexpected error running accoreconsole: {e}")
        return None
    finally:
        if os.path.exists(scr_path):
            os.remove(scr_path)


if __name__ == "__main__":
    test_dwg = r"C:\Users\kabdu\OneDrive\Desktop\Data & Telephone\DWG\P05-EL-619-F1.dwg"
    if os.path.exists(test_dwg):
        result = plot_dwg_to_pdf(test_dwg)
        if result:
            print(f"\nTest Plot Succeeded: {result}")
    else:
        print(f"Test DWG not found: {test_dwg}")
