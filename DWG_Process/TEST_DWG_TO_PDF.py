from DWG_TO_PDF import plot_dwg_to_pdf

dwg_file = r"C:\Users\kabdu\OneDrive\Desktop\AVIS\DATA\25172  DGCL Capella Hotel\19. ICT\SLD\DG-NCD-405-0000-WME-DWG-IT-800-0000014_DWG(00).dwg"

pdf_path = plot_dwg_to_pdf(dwg_file)

print("PDF created at:", pdf_path)
