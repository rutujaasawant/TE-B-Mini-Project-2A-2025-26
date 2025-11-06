import fitz
import re
import json
import os
import pandas as pd
from django.conf import settings
from uuid import uuid4
import PyPDF2

# --- your parsing helpers (same as before) ---

def parse_grading_system(text):
    ranges = []
    marks_line = None
    grade_line = None

    for line in text.splitlines():
        if line.strip().startswith("MARKS"):
            marks_line = line
        elif line.strip().startswith("GRADE") and not line.strip().startswith("GRADE POINT"):
            grade_line = line

    if not marks_line or not grade_line:
        return []

    marks_ranges = re.findall(r"(\d+\.?\d*)\s*to\s*(\d+\.?\d*)", marks_line)
    grades = grade_line.split(":")[1].split()

    for (low, high), grade in zip(marks_ranges, grades):
        ranges.append((float(low), float(high), grade))

    return ranges

def get_grade(total, grading_rules):
    for low, high, grade in grading_rules:
        if low <= total <= high:
            return grade
    return None

def extract_paper_mapping(doc):
    mapping = {}
    pattern = re.compile(r"([A-Z0-9]{3,})\s*[-–]\s*([A-Za-z0-9\s\(\)/&\.\-]+?)(?::|\n|$)")
    
    for page_num in range(min(5, len(doc))):
        page_text = doc[page_num].get_text()
        for code, name in pattern.findall(page_text):
            mapping[code.strip()] = name.strip()
    
    return mapping

def parse_student_block(block, grading_rules, paper_names):
    lines = [l.strip() for l in block.split("\n") if l.strip()]
    if not lines:
        return None

    header_line = lines[0]
    m = re.match(r"(\d{7})\s+([A-Z\s/]+?)\s+\|(.+?)\|\s*(Successful|Unsuccessful)", header_line)
    if not m:
        return None

    seat_no = m.group(1).strip()
    name = m.group(2).strip()
    result = m.group(4).strip()

    codes1 = re.findall(r'\|([A-Z0-9]{3,})\s', header_line)

    totals1 = []
    if len(lines) > 1:
        totals1 = [int(nums[-1]) for nums in [re.findall(r'\d+', seg) for seg in lines[1].split("|")[1:]] if nums]

    codes2, totals2 = [], []
    for i, line in enumerate(lines):
        if re.search(r"\(\d+\)\w+", line) or re.search(r'\|\s*[A-Z0-9]{3,}\s', line):
            codes2 = re.findall(r'\|([A-Z0-9]{3,})\s', line)
            if i + 1 < len(lines):
                totals2 = [int(nums[-1]) for nums in [re.findall(r'\d+', seg) for seg in lines[i+1].split("|")[1:]] if nums]
            break

    papers = []
    for c, t in zip(codes1, totals1):
        papers.append({
            "paper_code": c,
            "paper_name": paper_names.get(c, "Unknown"),
            "total": t,
            "grade": get_grade(t, grading_rules)
        })
    for c, t in zip(codes2, totals2):
        papers.append({
            "paper_code": c,
            "paper_name": paper_names.get(c, "Unknown"),
            "total": t,
            "grade": get_grade(t, grading_rules)
        })

    sgpi = None
    if result.lower() == "successful":
        for line in reversed(lines):
            m = re.search(r"\b(\d+\.\d+)\b\s+--", line)
            if m:
                sgpi = m.group(1)
                break

    return {
        "seat_no": seat_no,
        "name": name,
        "result": result,
        "sgpi": sgpi,
        "papers": papers
    }

def extract_result(file=None):
    # The directory where files will be saved, inside your MEDIA_ROOT
    upload_dir_name = "uploads"
    upload_dir = os.path.join(settings.MEDIA_ROOT, upload_dir_name)
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate a unique filename
    file_id = uuid4()
    
    # Save the uploaded PDF
    pdf_path = os.path.join(upload_dir, f"{file_id}.pdf")
    with open(pdf_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

    # Define filesystem paths for output files
    json_path = os.path.join(upload_dir, f"{file_id}.json")
    excel_path = os.path.join(upload_dir, f"{file_id}.xlsx")
    
    # Extract results
    with fitz.open(pdf_path) as doc:
        paper_names = extract_paper_mapping(doc)
        full_text = "\n".join(page.get_text() for page in doc)
        grading_rules = parse_grading_system(full_text)

    blocks = re.split(r"(?=\n\s*\d{7}\s)", full_text)
    results, rows = [], []

    for block in blocks:
        student_data = parse_student_block(block, grading_rules, paper_names)
        if student_data:
            results.append(student_data)
            
            row = {
                "Seat No": student_data["seat_no"],
                "Name": student_data["name"],
                "Result": student_data["result"],
                "SGPI": student_data["sgpi"]
            }
            for i, paper in enumerate(student_data["papers"], start=1):
                row[f"Paper {i} Code"] = paper["paper_code"]
                row[f"Paper {i} Name"] = paper["paper_name"]
                row[f"Paper {i} Marks"] = paper["total"]
                row[f"Paper {i} Grade"] = paper["grade"]
            rows.append(row)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save Excel
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)

    json_url = os.path.join(settings.MEDIA_URL, upload_dir_name, f"{file_id}.json").replace("\\", "/")
    excel_url = os.path.join(settings.MEDIA_URL, upload_dir_name, f"{file_id}.xlsx").replace("\\", "/")

    return results, json_url, excel_url

# --- PDF Percentage Analysis Helpers ---

def analyze_pdf_percentage(file):
    """
    Main function for API.
    Accepts Django UploadedFile.
    Returns results, json_url, excel_url.
    """
    # Step 1: Save uploaded PDF
    upload_dir_name = "uploads"
    upload_dir = os.path.join(settings.MEDIA_ROOT, upload_dir_name)
    os.makedirs(upload_dir, exist_ok=True)

    file_id = uuid4()
    pdf_path = os.path.join(upload_dir, f"{file_id}.pdf")
    with open(pdf_path, "wb") as f:
        for chunk in file.chunks():
            f.write(chunk)

    # Step 2: Extract text
    extracted_text = extract_text_from_pdf(pdf_path)
    if "error" in extracted_text.lower():
        raise ValueError(f"PDF extraction error: {extracted_text}")

    # Step 3: Parse subjects - NOW RETURNS BOTH MARKS AND NAMES
    total_marks_map, subject_names_map, subject_type_map = parse_subject_structure(extracted_text)
    if not total_marks_map:
        raise ValueError("Could not parse subjects from PDF.")

    num_subjects = len(total_marks_map)

    # Step 4: Parse students
    students = parse_students(extracted_text, num_subjects)
    if not students:
        raise ValueError("No student data found in PDF.")

    # Step 5: Calculate percentages with subject names and types
    results, _ = calculate_percentages(students, total_marks_map, subject_names_map, subject_type_map)

    # Step 6: Save Excel with NO duplicate columns
    excel_path = os.path.join(upload_dir, f"{file_id}.xlsx")
    df = pd.DataFrame(results)

    # Remove duplicate columns if any exist
    df = df.loc[:, ~df.columns.duplicated()].copy()

    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Write Results sheet
            df.to_excel(writer, sheet_name='Results', index=False)
            
            # Write Subject Structure sheet
            subject_data = []
            for code in total_marks_map.keys():
                subject_data.append({
                    'Subject Code': code,
                    'Subject Name': subject_names_map[code],
                    'Subject Type': subject_type_map[code].capitalize(),
                    'Maximum Marks': total_marks_map[code]
                })
            
            subject_df = pd.DataFrame(subject_data)
            subject_df.to_excel(writer, sheet_name='Subject Structure', index=False)
            
            # Get worksheets for formatting
            workbook = writer.book
            results_sheet = writer.sheets['Results']
            subject_sheet = writer.sheets['Subject Structure']
            
            # Auto-adjust column widths
            for sheet in [results_sheet, subject_sheet]:
                for column_cells in sheet.columns:
                    max_length = 0
                    column_letter = column_cells[0].column_letter
                    
                    for cell in column_cells:
                        try:
                            if cell.value:
                                max_length = max(max_length, len(str(cell.value)))
                        except:
                            pass
                    
                    adjusted_width = max(12, min(max_length + 3, 45))
                    sheet.column_dimensions[column_letter].width = adjusted_width
            
            # Freeze header row
            results_sheet.freeze_panes = 'A2'
            subject_sheet.freeze_panes = 'A2'

    except Exception as e:
        raise ValueError(f"Excel generation error: {e}")

    # Step 7: Save JSON
    json_path = os.path.join(upload_dir, f"{file_id}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    # Step 8: Return URLs
    json_url = os.path.join(settings.MEDIA_URL, upload_dir_name, f"{file_id}.json").replace("\\", "/")
    excel_url = os.path.join(settings.MEDIA_URL, upload_dir_name, f"{file_id}.xlsx").replace("\\", "/")

    return results, json_url, excel_url

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ""
            # print(f"Extracted text: {text[:50000]}...")  # Print first 500 characters for debugging
                   
    except Exception as e:
        return f"An error occurred: {e}"
       
    return text

def parse_subject_structure(text):
    """Parse subjects and identify theory vs practical"""
    university_pattern = r'University\s+of\s+Mumbai'
    match = re.search(university_pattern, text, re.IGNORECASE)
    
    if not match:
        for alt in [r'OFFICE\s+REGISTER', r'CBCS.*Engineering']:
            alt_match = re.search(alt, text, re.IGNORECASE)
            if alt_match:
                start_pos = alt_match.start()
                break
        else:
            return {}, {}, {}
    else:
        start_pos = match.start()
    
    subject_section = text[start_pos:start_pos + 5000]
    
    total_marks_map = {}
    subject_names_map = {}
    subject_type_map = {}  # NEW: Track if subject is theory or practical
    subject_order = []
    
    pattern = r'([A-Z0-9]{5,6}(?:\s+[A-Z]{2,3})?)(?:\s*[‐\-–]\s*)([^:]+?):\s+.*?(\d{2,3})/0'
    matches = re.findall(pattern, subject_section, re.MULTILINE)
    
    for subject_code, subject_name, max_marks in matches:
        subject_code = subject_code.strip()
        subject_name = subject_name.strip()
        
        if subject_code not in total_marks_map:
            total_marks_map[subject_code] = int(max_marks)
            subject_names_map[subject_code] = subject_name
            
            # Check if it's a theory subject (numeric code) or practical (FEC, FEL, etc.)
            if subject_code.isdigit():
                subject_type_map[subject_code] = 'theory'
            else:
                subject_type_map[subject_code] = 'practical'
            
            subject_order.append(subject_code)
    
    ordered_marks = {code: total_marks_map[code] for code in subject_order}
    ordered_names = {code: subject_names_map[code] for code in subject_order}
    ordered_types = {code: subject_type_map[code] for code in subject_order}
    
    return ordered_marks, ordered_names, ordered_types

def extract_marks_from_cell(cell):
    """Extract marks: handles AA, --, 7F, 10E, 23F, and SPACES in numbers"""
    cell = cell.strip()
    
    # Remove all spaces from the cell first to handle "6 1" -> "61"
    cell_no_space = cell.replace(' ', '')
    
    # Pattern 1: Three values (Theory, ClassTest, Total) - e.g., "67 19 86" or "48 13 6 1"
    # First try without spaces
    match1 = re.match(r'^(\d+)\s+(\d+)\s+(\d+)$', cell)
    if match1:
        theory = int(match1.group(1))
        class_test = int(match1.group(2))
        total = int(match1.group(3))
        return {'theory': theory, 'class_test': class_test, 'total': total}
    
    # Try matching with potential spaces in numbers - e.g., "48 13 6 1" -> "48 13 61"
    match1b = re.match(r'^(\d+(?:\s+\d)*)\s+(\d+(?:\s+\d)*)\s+(\d+(?:\s+\d)*)$', cell)
    if match1b:
        theory = int(match1b.group(1).replace(' ', ''))
        class_test = int(match1b.group(2).replace(' ', ''))
        total = int(match1b.group(3).replace(' ', ''))
        return {'theory': theory, 'class_test': class_test, 'total': total}
    
    # Pattern 2: Theory with letter grade and total - e.g., "67F 19 86"
    match2 = re.match(r'^(\d+)[A-Z]\s+(\d+)\s+(\d+)$', cell)
    if match2:
        theory = int(match2.group(1))
        class_test = int(match2.group(2))
        total = int(match2.group(3))
        return {'theory': theory, 'class_test': class_test, 'total': total}
    
    # Pattern 3: Single value with letter grade - e.g., "67E"
    match3 = re.match(r'^(\d+)[A-Z]$', cell_no_space)
    if match3:
        total = int(match3.group(1))
        return {'total': total}
    
    # Pattern 4: Practical marks with spaces - e.g., "-- 22 4 5" -> "-- 22 45"
    match4 = re.match(r'^[‐\-–]+\s+(\d+(?:\s+\d)*)\s+(\d+(?:\s+\d)*)$', cell)
    if match4:
        val1 = int(match4.group(1).replace(' ', ''))
        val2 = int(match4.group(2).replace(' ', ''))
        return {'total': val2}
    
    # Pattern 5: Practical marks simple - e.g., "-- 22 22" or "-- 22"
    match5 = re.match(r'^[‐\-–]+\s+(\d+)(?:\s+(\d+))?$', cell)
    if match5:
        val1 = int(match5.group(1))
        val2 = int(match5.group(2)) if match5.group(2) else val1
        return {'total': val2}
    
    # Pattern 6: Practical with multiple values and dashes - e.g., "-- 22 -- 23"
    match6 = re.match(r'^[‐\-–]+\s+(\d+)\s+[‐\-–]+\s+(\d+)$', cell)
    if match6:
        val1 = int(match6.group(1))
        val2 = int(match6.group(2))
        return {'total': val1 + val2}  # Sum both values
    
    return None

def parse_students(text, num_subjects):
    """Parse students - returns mark dictionaries instead of integers"""
    students = []
    
    match = re.search(r'University\s+of\s+Mumbai', text, re.IGNORECASE)
    if match:
        text = text[match.start():]
    
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        student_match = re.search(r'(\d{7})\s+(/\s+)?([A-Z][A-Z\s]+?)\s+\|', line)
        
        if student_match:
            seat_no = student_match.group(1)
            name = student_match.group(3).strip()
            
            marks = []
            for j in range(i, min(i + 20, len(lines))):
                cells = lines[j].split('|')
                for cell in cells:
                    mark_info = extract_marks_from_cell(cell)
                    if mark_info is not None and len(marks) < num_subjects:
                        marks.append(mark_info)
                if len(marks) >= num_subjects:
                    break
            
            if len(marks) >= num_subjects:
                students.append({
                    'seat_no': seat_no,
                    'name': name,
                    'marks': marks[:num_subjects]
                })
        
        i += 1
    
    return students

def calculate_percentages(students, total_marks_map, subject_names_map, subject_type_map):
    """Calculate percentages - ensure no duplicate column names"""
    total_maximum_marks = sum(total_marks_map.values())
    results = []
    subject_codes = list(total_marks_map.keys())
    
    for student in students:
        name = student['name']
        marks_data = student['marks']
        
        result = {}
        result['Name'] = name
        
        total_obtained = 0
        used_column_names = set()  # Track used column names to avoid duplicates
        
        for i, subject_code in enumerate(subject_codes):
            subject_name = subject_names_map.get(subject_code, subject_code)
            subject_type = subject_type_map.get(subject_code, 'practical')
            
            if i < len(marks_data):
                mark_info = marks_data[i]
                
                if subject_type == 'theory' and 'theory' in mark_info and 'class_test' in mark_info:
                    # Theory subject with breakdown
                    theory_col = f"{subject_name} Theory"
                    ct_col = f"{subject_name} Class Test"
                    total_col = f"{subject_name} Total"
                    
                    # Only add if not already present
                    if theory_col not in used_column_names:
                        result[theory_col] = mark_info['theory']
                        used_column_names.add(theory_col)
                    
                    if ct_col not in used_column_names:
                        result[ct_col] = mark_info['class_test']
                        used_column_names.add(ct_col)
                    
                    if total_col not in used_column_names:
                        result[total_col] = mark_info['total']
                        used_column_names.add(total_col)
                    
                    total_obtained += mark_info['total']
                else:
                    # Practical subject
                    if subject_name not in used_column_names:
                        result[subject_name] = mark_info.get('total', 0)
                        used_column_names.add(subject_name)
                        total_obtained += mark_info.get('total', 0)
            else:
                # Missing data
                if subject_type == 'theory':
                    theory_col = f"{subject_name} Theory"
                    ct_col = f"{subject_name} Class Test"
                    total_col = f"{subject_name} Total"
                    
                    if theory_col not in used_column_names:
                        result[theory_col] = 0
                        used_column_names.add(theory_col)
                    if ct_col not in used_column_names:
                        result[ct_col] = 0
                        used_column_names.add(ct_col)
                    if total_col not in used_column_names:
                        result[total_col] = 0
                        used_column_names.add(total_col)
                else:
                    if subject_name not in used_column_names:
                        result[subject_name] = 0
                        used_column_names.add(subject_name)
        
        percentage = (total_obtained / total_maximum_marks) * 100
        result['Percentage'] = round(percentage, 2)
        
        results.append(result)
    
    return results, total_maximum_marks

