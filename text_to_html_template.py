"""
Simple Medical Note to HTML Converter
Parses SOAP-format notes into collapsible HTML sections with bullet points.
"""

import re
from pathlib import Path
from typing import Optional


def parse_sections(text: str) -> dict:
    """Parse text into main SOAP sections."""
    sections = {}
    
    # Extract patient info from header
    patient_match = re.search(
        r'Pt:\s*(.+?)(?=\s+S:|$)', 
        text, 
        re.DOTALL | re.IGNORECASE
    )
    if patient_match:
        sections['Patient Info'] = patient_match.group(1).strip()
    
    # Extract S section
    s_match = re.search(r'\bS:\s*(.*?)(?=\bO:|$)', text, re.DOTALL | re.IGNORECASE)
    if s_match:
        sections['Subjective'] = s_match.group(1).strip()
    
    # Extract O section
    o_match = re.search(r'\bO:\s*(.*?)(?=\bA/P:|A:|$)', text, re.DOTALL | re.IGNORECASE)
    if o_match:
        sections['Objective'] = o_match.group(1).strip()
    
    # Extract A/P section
    ap_match = re.search(r'\bA/P:\s*(.*?)$', text, re.DOTALL | re.IGNORECASE)
    if not ap_match:
        ap_match = re.search(r'\bA:\s*(.*?)$', text, re.DOTALL | re.IGNORECASE)
    if ap_match:
        sections['Assessment/Plan'] = ap_match.group(1).strip()
    
    return sections


def format_text(text: str) -> str:
    """Format text as a simple paragraph."""
    return f"<p>{text}</p>"


def extract_ejection_fraction(text: str) -> list:
    """
    Extract Ejection Fraction values from text.
    Looks for: EF, LVEF, RVEF, VEF with associated values and dates.
    """
    results = []
    
    # Date patterns to look for after EF values
    date_pattern = r'''
        (?:
            (?:on|dated|from|in)\s+)?
        (
            \d{1,2}/\d{1,2}/\d{2,4}|           # MM/DD/YYYY or MM/DD/YY
            \d{4}-\d{2}-\d{2}|                  # YYYY-MM-DD
            (?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|  # Month DD, YYYY
            (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}|  # Mon DD, YYYY
            \d{4}                               # Just year
        )
    '''
    
    # Pattern to match EF abbreviations with values and optional dates
    ef_pattern = r'''
        (?P<type>LVEF|RVEF|VEF|EF)          # EF type abbreviation
        \s*                                   # optional space
        (?:of\s+|at\s+|is\s+|was\s+|=\s*|:\s*)?     # optional connector words
        (?P<value>
            \d{1,2}                           # first number (1-2 digits)
            (?:\s*[-–to]+\s*\d{1,2})?        # optional range (e.g., -25, to 25)
        )
        \s*
        (?P<unit>%|percent)?                 # unit (% or percent)
    '''
    
    matches = re.finditer(ef_pattern, text, re.IGNORECASE | re.VERBOSE)
    
    for match in matches:
        ef_type = match.group('type').upper()
        value = match.group('value').strip()
        unit = match.group('unit') or '%'
        if unit.lower() == 'percent':
            unit = '%'
        
        # Normalize the value (replace 'to' with '-')
        value = re.sub(r'\s*to\s*', '-', value)
        value = re.sub(r'\s*–\s*', '-', value)  # en-dash
        
        # Look for date after the match (within next 100 chars)
        after_text = text[match.end():match.end()+100]
        date_match = re.search(date_pattern, after_text, re.IGNORECASE | re.VERBOSE)
        date_str = date_match.group(1).strip() if date_match else None
        
        # Full name mapping
        type_names = {
            'EF': 'Ejection Fraction',
            'LVEF': 'Left Ventricular EF',
            'RVEF': 'Right Ventricular EF',
            'VEF': 'Ventricular EF'
        }
        
        results.append({
            'type': ef_type,
            'full_name': type_names.get(ef_type, ef_type),
            'value': f"{value}{unit}",
            'date': date_str
        })
    
    # Also look for "ejection fraction" written out with optional date
    written_pattern = r'''
        (?P<qualifier>left\s+ventricular\s+|right\s+ventricular\s+|ventricular\s+)?
        ejection\s+fraction
        (?:\s+(?:documented\s+)?(?:at|of|is|was|=|:))?\s*
        (?P<value>\d{1,2}(?:\s*[-–to]+\s*\d{1,2})?)
        \s*
        (?P<unit>%|percent)?
    '''
    
    written_matches = re.finditer(written_pattern, text, re.IGNORECASE | re.VERBOSE)
    
    for match in written_matches:
        qualifier = (match.group('qualifier') or '').strip().lower()
        value = match.group('value').strip()
        unit = match.group('unit') or '%'
        if unit.lower() == 'percent':
            unit = '%'
        
        value = re.sub(r'\s*to\s*', '-', value)
        value = re.sub(r'\s*–\s*', '-', value)
        
        # Look for date after the match
        after_text = text[match.end():match.end()+100]
        date_match = re.search(date_pattern, after_text, re.IGNORECASE | re.VERBOSE)
        date_str = date_match.group(1).strip() if date_match else None
        
        if 'left' in qualifier:
            ef_type = 'LVEF'
            full_name = 'Left Ventricular EF'
        elif 'right' in qualifier:
            ef_type = 'RVEF'
            full_name = 'Right Ventricular EF'
        else:
            ef_type = 'EF'
            full_name = 'Ejection Fraction'
        
        # Check if this value wasn't already captured
        if not any(r['value'] == f"{value}{unit}" and r['type'] == ef_type for r in results):
            results.append({
                'type': ef_type,
                'full_name': full_name,
                'value': f"{value}{unit}",
                'date': date_str
            })
    
    return results


def parse_numbered_problems(text: str) -> list:
    """Parse numbered problems from A/P section."""
    problems = []
    
    # Split by numbered items at start of line or after period+space
    # Pattern: number followed by period, then title with colon
    parts = re.split(r'(?:^|\.\s+)(\d{1,2})\.\s+([A-Z][^:]+):\s*', text, flags=re.MULTILINE)
    
    # parts will be: [intro, num1, title1, content1, num2, title2, content2, ...]
    if len(parts) > 1:
        # Skip the intro part (index 0), then process in groups of 3
        i = 1
        while i + 2 <= len(parts):
            num = parts[i]
            title = parts[i + 1] if i + 1 < len(parts) else ""
            content = parts[i + 2] if i + 2 < len(parts) else ""
            
            if num and title:
                problems.append({
                    'number': num,
                    'title': title.strip(),
                    'content': content.strip() if content else ""
                })
            i += 3
    
    return problems


def generate_html(sections: dict, raw_text: str = "") -> str:
    """Generate simple HTML with collapsible sections."""
    
    # Extract Ejection Fraction values from entire text
    ef_values = extract_ejection_fraction(raw_text)
    
    # Build EF findings section
    ef_html = ""
    if ef_values:
        ef_items = ""
        for ef in ef_values:
            date_str = f" <span class='ef-date'>({ef['date']})</span>" if ef.get('date') else ""
            ef_items += f"<p><strong>{ef['type']}</strong>: <span class='ef-value'>{ef['value']}</span>{date_str}</p>"
        
        ef_html = f"""
    <div class="ef-findings">
        <h3>Ejection Fraction</h3>
        {ef_items}
    </div>"""
    
    # Build patient header
    patient_html = ""
    if 'Patient Info' in sections:
        info = sections['Patient Info']
        # Extract key fields
        name = re.search(r'^([A-Za-z\s]+?)(?=\s+DOB|\s*$)', info)
        dob = re.search(r'DOB:\s*(\S+)', info)
        age = re.search(r'Age[:\s]+(\d+)', info)
        mrn = re.search(r'MRN:\s*(\S+)', info)
        visit = re.search(r'Visit Date:\s*(\S+)', info)
        attending = re.search(r'Attending[^:]*:\s*([^D]+?)(?=DOB|$)', info) or \
                   re.search(r'Attending[^:]*:\s*(.+?)$', info)
        
        patient_html = f"""
    <div class="patient-header">
        <h2>{name.group(1).strip() if name else 'Patient'}</h2>
        <p><strong>DOB:</strong> {dob.group(1) if dob else 'N/A'} | <strong>Age:</strong> {age.group(1) if age else 'N/A'} | <strong>MRN:</strong> {mrn.group(1) if mrn else 'N/A'}</p>
        <p><strong>Visit Date:</strong> {visit.group(1) if visit else 'N/A'} | <strong>Attending:</strong> {attending.group(1).strip() if attending else 'N/A'}</p>
    </div>"""
    
    # Build main sections
    section_html = ""
    
    for section_name in ['Subjective', 'Objective', 'Assessment/Plan']:
        if section_name not in sections:
            continue
        
        content = sections[section_name]
        
        if section_name == 'Assessment/Plan':
            # Parse numbered problems
            problems = parse_numbered_problems(content)
            if problems:
                problem_details = ""
                for p in problems:
                    problem_details += f"""
        <details>
            <summary>{p['number']}. {p['title']}</summary>
            <div class="content">
                {format_text(p['content'])}
            </div>
        </details>"""
                
                section_html += f"""
    <details open>
        <summary>{section_name}</summary>
        <div class="content">
            {problem_details}
        </div>
    </details>"""
            else:
                section_html += f"""
    <details open>
        <summary>{section_name}</summary>
        <div class="content">
            {format_text(content)}
        </div>
    </details>"""
        else:
            section_html += f"""
    <details open>
        <summary>{section_name}</summary>
        <div class="content">
            {format_text(content)}
        </div>
    </details>"""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Patient Record</title>
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #f9f9f9;
        }}
        
        .patient-header {{
            background: #e3f2fd;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        
        .patient-header h2 {{
            margin: 0 0 10px 0;
            color: #1565c0;
        }}
        
        .patient-header p {{
            margin: 5px 0;
        }}
        
        .ef-findings {{
            background: #fff3e0;
            border: 2px solid #ff9800;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        
        .ef-findings h3 {{
            margin: 0 0 10px 0;
            color: #e65100;
        }}
        
        .ef-findings p {{
            margin: 5px 0;
        }}
        
        .ef-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #d84315;
        }}
        
        .ef-date {{
            font-size: 0.9em;
            color: #666;
        }}
        
        details {{
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
        }}
        
        summary {{
            padding: 12px 15px;
            font-weight: 600;
            cursor: pointer;
            background: #1565c0;
            color: white;
            border-radius: 3px;
        }}
        
        summary:hover {{
            background: #1976d2;
        }}
        
        details details summary {{
            background: #43a047;
        }}
        
        details details summary:hover {{
            background: #4caf50;
        }}
        
        details details details summary {{
            background: #7b1fa2;
        }}
        
        .content {{
            padding: 15px;
        }}
        
        .content p {{
            margin: 0;
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
    {patient_html}
    {ef_html}
    {section_html}
</body>
</html>"""


def convert_note_to_html(note_text: str, output_path: Optional[str] = None) -> str:
    """Convert medical note text to HTML."""
    sections = parse_sections(note_text)
    html = generate_html(sections, raw_text=note_text)
    
    if output_path:
        Path(output_path).write_text(html, encoding='utf-8')
        print(f"HTML saved to: {output_path}")
    
    return html


if __name__ == "__main__":
    import sys
    import os
    
    script_name = os.path.basename(__file__)
    
    if len(sys.argv) < 2:
        print(f"Usage: python {script_name} <input_file> [output_file]")
        print(f"       python {script_name} --stdin [output_file]")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if input_arg == "--stdin":
        print("Paste medical note text (Ctrl+D or Ctrl+Z to finish):")
        note_text = sys.stdin.read()
    else:
        note_text = Path(input_arg).read_text(encoding='utf-8')
    
    html = convert_note_to_html(note_text, output_file)
    
    if not output_file:
        print(html)
