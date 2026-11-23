import csv
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import fitz
from openpyxl import Workbook, load_workbook


BASE_DIR = Path(__file__).parent

CSV_FILE = BASE_DIR / "employees.csv"
JSON_FILE = BASE_DIR / "employees.json"
HIGH_EARNERS_FILE = BASE_DIR / "high_earners.csv"
EXCEL_FILE = BASE_DIR / "salary.xlsx"
PDF_FILE = BASE_DIR / "sample.pdf"


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



def normalize_salary(salary: Union[int, str]) -> int:

    return int(salary)



def read_employees(path: Union[str, Path]) -> List[dict]:

    employees = []

    with open(path, "r", newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:

            employee = {
                "name": row["name"],
                "department": row["department"],
                "salary": normalize_salary(row["salary"])
            }

            employees.append(employee)

    return employees



def write_json(
    employees: List[dict],
    path: Union[str, Path]
) -> None:

    with open(path, "w", encoding="utf-8") as file:

        json.dump(employees, file, indent=4)



def write_high_earners(
    employees: List[dict],
    path: Union[str, Path]
) -> int:

    high_earners = [
        emp for emp in employees
        if emp["salary"] > 50000
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:

        fieldnames = ["name", "department", "salary"]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(high_earners)

    return len(high_earners)



def create_excel(
    employees: List[dict],
    path: Union[str, Path]
) -> None:

    workbook = Workbook()

    sheet = workbook.active
    sheet.title = "Employee Salaries"

    sheet.append(["Name", "Department", "Salary"])

    for emp in employees:

        sheet.append([
            emp["name"],
            emp["department"],
            emp["salary"]
        ])

    workbook.save(path)



def read_excel(path: Union[str, Path]) -> int:

    workbook = load_workbook(path)

    sheet = workbook.active

    row_count = sheet.max_row - 1

    workbook.close()

    return row_count



def extract_pdf_text(path: Union[str, Path]) -> str:

    with fitz.open(path) as document:

        if len(document) == 0:
            return ""

        page = document[0]

        text = page.get_text()

        return text[:100]


def read_file(
    path: Union[str, Path]
) -> Optional[Union[dict, list]]:

    file_path = Path(path)

    try:

        with open(file_path, "r", encoding="utf-8") as file:

            return json.load(file)

    except FileNotFoundError:

        logger.error("File not found: %s", file_path.name)

    except json.JSONDecodeError:

        logger.error("Invalid JSON format: %s", file_path.name)

    return None


def find_employee(
    name: str,
    employees: List[dict]
) -> Optional[dict]:

    for emp in employees:

        if emp["name"].lower() == name.lower():

            return emp

    return None


def main() -> None:

    employees = read_employees(CSV_FILE)

    print(f"Rows read: {len(employees)}")

    write_json(employees, JSON_FILE)

    json_data = read_file(JSON_FILE)

    high_count = write_high_earners(
        employees,
        HIGH_EARNERS_FILE
    )

    print(f"High earners written: {high_count}")

    create_excel(employees, EXCEL_FILE)

    excel_rows = read_excel(EXCEL_FILE)

    print(
        f"salary.xlsx created and read back: {excel_rows} rows"
    )

    if PDF_FILE.exists():

        pdf_text = extract_pdf_text(PDF_FILE)

        print("PDF first 100 characters:")
        print(pdf_text)

    else:

        logger.warning("PDF not found: %s", PDF_FILE.name)

    read_file(BASE_DIR / "missing.txt")


    result = find_employee("Divya", employees)

    if result:
        print("Employee found:", result)



if __name__ == "__main__":

    main()

