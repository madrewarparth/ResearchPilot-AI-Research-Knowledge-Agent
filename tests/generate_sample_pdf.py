"""
Utility to generate a sample research PDF for testing and manual validation.
"""

from pathlib import Path
from pypdf import PdfWriter

def create_sample_research_pdf(output_path: Path):
    try:
        import fitz
        doc = fitz.open()
        
        # Page 1
        page1 = doc.new_page(width=612, height=792)
        text1 = (
            "Research Report on Renewable Energy Integration (2024)\n\n"
            "Executive Summary:\n"
            "This study investigates grid stabilization and storage dynamics across hybrid solar-wind installations.\n"
            "Perovskite solar cell efficiency reached 25.7% in laboratory testing environments under standard irradiance.\n"
            "The operational capacity factor of newly deployed wind turbines exceeded 42 percent in coastal zones.\n\n"
            "Section 1: Solar Photovoltaic Advances\n"
            "Recent advancements in tandem perovskite-silicon architectures have overcome previous moisture degradation barriers.\n"
            "Thermal cycling tests over 1000 hours demonstrated 94 percent retention of initial photovoltage."
        )
        page1.insert_text((50, 70), text1, fontsize=11)

        # Page 2
        page2 = doc.new_page(width=612, height=792)
        text2 = (
            "Section 2: Wind Turbine Aerodynamics & Grid Storage\n\n"
            "Active blade pitch optimization dynamically mitigates aerodynamic blade stress and fatigue during severe wind gusts.\n"
            "Grid battery storage using lithium-iron-phosphate packs exhibited an 8.2% degradation in capacity after 1000 charge cycles.\n\n"
            "Limitations:\n"
            "The authors acknowledge that winter sub-zero temperatures were not tested in the current simulation model.\n"
            "Further empirical data across offshore installations is required."
        )
        page2.insert_text((50, 70), text2, fontsize=11)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(output_path))
        doc.close()
        return True
    except Exception:
        # Fallback using pypdf blank page
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        writer.add_blank_page(width=612, height=792)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            writer.write(f)
        return True

if __name__ == "__main__":
    out = Path("data/sample_renewable_study.pdf")
    create_sample_research_pdf(out)
    print(f"Sample PDF created at {out}")
