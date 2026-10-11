from pathlib import Path

from openpyxl import load_workbook

from app.tools import create_excel


def test_create_excel_writes_file(tmp_path: Path) -> None:
    output = tmp_path / "report.xlsx"
    rows = [{"name": "alpha", "score": 10}, {"name": "beta", "score": 20}]

    result = create_excel(str(output), rows)

    assert output.exists()
    assert result["rows_written"] == 2

    workbook = load_workbook(output)
    worksheet = workbook.active
    assert worksheet["A1"].value == "name"
    assert worksheet["B1"].value == "score"
    assert worksheet["A2"].value == "alpha"
