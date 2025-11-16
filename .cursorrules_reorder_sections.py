#!/usr/bin/env python3
"""
Reorder and renumber .cursorrules sections.

This script:
1. Ignores the Table of Contents (doesn't parse it)
2. Auto-detects sections by separator lines + titles
3. Renumbers sections sequentially based on their order in the file
4. Regenerates the Table of Contents to match

Usage:
    python .cursorrules_reorder_sections.py           # Apply changes
    python .cursorrules_reorder_sections.py --dry-run # Preview only
"""

import re
import sys
from pathlib import Path


def find_toc_boundaries(lines):
    """Find start and end of Table of Contents."""
    toc_start = None
    toc_end = None
    
    for i, line in enumerate(lines):
        if line.strip() == 'TABLE OF CONTENTS':
            toc_start = i - 1  # Include separator line above
        elif toc_start is not None:
            # Look for the first section header after ToC (separator + number. TITLE + separator)
            if (line == '=============================================================================' and 
                i + 2 < len(lines) and
                re.match(r'^\d+\. [A-Z]', lines[i + 1]) and
                lines[i + 2] == '============================================================================='):
                toc_end = i  # End right before first section
                break
    
    return toc_start, toc_end


def parse_sections(lines, start_idx):
    """Parse all sections starting from start_idx (after ToC)."""
    sections = []
    i = start_idx
    
    while i < len(lines):
        # Look for section: separator + title + separator
        if (lines[i] == '=============================================================================' and 
            i + 2 < len(lines) and
            lines[i + 2] == '============================================================================='):
            
            title_line = lines[i + 1]
            match = re.match(r'^(\d+)\. (.+)$', title_line)
            if match:
                old_num = int(match.group(1))
                title = match.group(2)
                
                # Find end of section (next section or EOF)
                section_start = i
                section_end = len(lines)
                
                for j in range(i + 3, len(lines)):
                    if (lines[j] == '=============================================================================' and 
                        j + 2 < len(lines) and
                        lines[j + 2] == '============================================================================='):
                        section_end = j
                        break
                
                # Parse subsections within this section
                subsections = parse_subsections(lines[section_start:section_end])
                
                sections.append({
                    'old_num': old_num,
                    'title': title,
                    'start': section_start,
                    'end': section_end,
                    'subsections': subsections,
                    'lines': lines[section_start:section_end]
                })
                
                i = section_end
                continue
        i += 1
    
    return sections


def parse_subsections(section_lines):
    """Parse ## N.M subsections and ### N.M.K sub-subsections."""
    subsections = []
    current_subsection = None
    
    for line in section_lines:
        # Subsection: ## N.M Title
        match = re.match(r'^## (\d+)\.(\d+) (.+)$', line)
        if match:
            current_subsection = {
                'old_num': f"{match.group(1)}.{match.group(2)}",
                'title': match.group(3),
                'subsubsections': []
            }
            subsections.append(current_subsection)
            continue
        
        # Sub-subsection: ### N.M.K Title
        match = re.match(r'^### (\d+)\.(\d+)\.(\d+) (.+)$', line)
        if match and current_subsection:
            current_subsection['subsubsections'].append({
                'old_num': f"{match.group(1)}.{match.group(2)}.{match.group(3)}",
                'title': match.group(4)
            })
    
    return subsections


def renumber_sections(sections):
    """Renumber sections sequentially starting from 1."""
    for new_num, section in enumerate(sections, start=1):
        section['new_num'] = new_num
        
        # Renumber subsections
        for sub_idx, subsection in enumerate(section['subsections'], start=1):
            subsection['new_num'] = f"{new_num}.{sub_idx}"
            
            # Renumber sub-subsections
            for subsub_idx, subsubsection in enumerate(subsection['subsubsections'], start=1):
                subsubsection['new_num'] = f"{new_num}.{sub_idx}.{subsub_idx}"


def renumber_section_content(section):
    """Renumber all headers within a section's content."""
    new_lines = []
    old_num = section['old_num']
    new_num = section['new_num']
    
    for line in section['lines']:
        # Section title
        if line == f"{old_num}. {section['title']}":
            line = f"{new_num}. {section['title']}"
        
        # Subsections ## N.M (match by title, not number)
        elif line.startswith('## '):
            match = re.match(r'^## \d+\.\d+ (.+)$', line)
            if match:
                title = match.group(1)
                for subsection in section['subsections']:
                    if subsection['title'] == title:
                        line = f"## {subsection['new_num']} {title}"
                        break
        
        # Sub-subsections ### N.M.K (match by title, not number)
        elif line.startswith('### '):
            match = re.match(r'^### \d+\.\d+\.\d+ (.+)$', line)
            if match:
                title = match.group(1)
                for subsection in section['subsections']:
                    for subsubsection in subsection['subsubsections']:
                        if subsubsection['title'] == title:
                            line = f"### {subsubsection['new_num']} {title}"
                            break
        
        # Sub-sub-subsections #### N.M.K.L (match by title, not number)
        elif line.startswith('#### '):
            match = re.match(r'^#### \d+\.\d+\.\d+\.\d+ (.+)$', line)
            if match:
                title = match.group(1)
                # Just renumber based on parent subsection
                line = re.sub(r'^#### \d+\.\d+\.', f'#### {new_num}.', line)
        
        new_lines.append(line)
    
    return new_lines


def generate_toc(sections):
    """Generate Table of Contents."""
    toc = [
        '=============================================================================',
        'TABLE OF CONTENTS',
        '=============================================================================',
    ]
    
    for section in sections:
        toc.append(f"{section['new_num']}. {section['title']}")
        
        if section['subsections']:
            for subsection in section['subsections']:
                toc.append(f"  {subsection['new_num']} {subsection['title']}")
                
                if subsection['subsubsections']:
                    for subsubsection in subsection['subsubsections']:
                        toc.append(f"    {subsubsection['new_num']} {subsubsection['title']}")
        
        toc.append('')
    
    return toc


def main():
    # Check for dry-run flag
    dry_run = any(arg in ['--dry-run', '--dryrun', '-n'] for arg in sys.argv[1:])
    
    cursorrules_path = Path(__file__).parent / '.cursorrules'
    
    if not cursorrules_path.exists():
        print(f"Error: {cursorrules_path} not found")
        return 1
    
    # Read file
    content = cursorrules_path.read_text()
    lines = content.splitlines()
    
    # Find ToC boundaries
    toc_start, toc_end = find_toc_boundaries(lines)
    
    if toc_start is None or toc_end is None:
        print("Error: Could not find Table of Contents")
        return 1
    
    print(f"Found ToC at lines {toc_start+1}-{toc_end}")
    
    # Extract header (before ToC)
    header = lines[:toc_start]
    
    # Parse sections (after ToC)
    sections = parse_sections(lines, toc_end)
    
    print(f"\nFound {len(sections)} sections:")
    for section in sections:
        print(f"  {section['old_num']}. {section['title']} ({len(section['subsections'])} subsections)")
    
    # Renumber sections
    renumber_sections(sections)
    
    # Show what will change
    print("\nRenumbering plan:")
    for section in sections:
        if section['old_num'] != section['new_num']:
            print(f"  {section['old_num']} → {section['new_num']}. {section['title']}")
        else:
            print(f"  {section['new_num']}. {section['title']} (no change)")
    
    if dry_run:
        print("\n🔍 DRY RUN MODE - No changes written")
        return 0
    
    # Renumber content within each section
    renumbered_body = []
    for section in sections:
        renumbered_body.extend(renumber_section_content(section))
    
    # Generate new ToC
    toc = generate_toc(sections)
    
    # Build final output
    output = header + [''] + toc + [''] + renumbered_body
    
    # Write back
    cursorrules_path.write_text('\n'.join(output) + '\n')
    print(f"\n✅ Updated {cursorrules_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
