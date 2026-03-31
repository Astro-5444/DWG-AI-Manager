import subprocess
import os

def render_pdf_dual(pdf_path, output_root="test_output", dpi_high=400, dpi_low=50):

    gs_path = r"OCR\gs10.06.0\bin\gswin64c.exe"

    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    save_dir = os.path.join(output_root, pdf_name)
    os.makedirs(save_dir, exist_ok=True)

    high_path = os.path.join(save_dir, "High_Resolution.png")
    low_path  = os.path.join(save_dir, "Low_Resolution.png")

    # Check if images already exist
    if os.path.exists(high_path) and os.path.exists(low_path):
        return {"low": low_path, "high": high_path}

    # ---------- HIGH RES ----------
    command_high = [
        gs_path,
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=png16m",          
        f"-r{dpi_high}",
        "-dTextAlphaBits=1",
        "-dGraphicsAlphaBits=1",
        f"-sOutputFile={high_path}",
        pdf_path
    ]

    subprocess.run(command_high, check=True)

    # ---------- LOW RES ----------
    command_low = [
        gs_path,
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pnggray",
        f"-r{dpi_low}",
        "-dTextAlphaBits=1",
        "-dGraphicsAlphaBits=1",
        f"-sOutputFile={low_path}",
        pdf_path
    ]

    subprocess.run(command_low, check=True)

    return {"low": low_path, "high": high_path}


if __name__ == "__main__":
    render_pdf_dual(
        "43001-AJI-04-DWG-IC-BL1-210006-000.pdf",
        output_root="test_output",
        dpi_high=400,
        dpi_low=50
    )
