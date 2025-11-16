#!/usr/bin/env python3
"""
Reorder and renumber .cursorrules sections.

Distinguishes between:
- Section headers: # N. TITLE (preceded by # ===...===)
- Subsection headers: ## N.M TITLE
- Sub-subsection headers: ### N.M.K TITLE
- Regular numbered lists: # N. item (not preceded by separator)
"""

import re
from pathlib import Path


def parse_sections(lines):
    """Parse all sections, subsections, and sub-subsections with their content."""
    sections = []
    current_section = None
    current_subsection = None
    current_subsubsection = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a section header (preceded by separator)
        if i > 0 and lines[i-1].startswith('# ===='):
            match = re.match(r'^# (\d+)\. (.+)$', line)
            if match:
                current_section = {
                    'type': 'section',
                    'old_num': int(match.group(1)),
                    'title': match.group(2),
                    'subsections': [],
                    'content': []
                }
                sections.append(current_section)
                current_subsection = None
                current_subsubsection = None
                i += 1
                continue
        
        # Subsection: ## N.M TITLE
        match = re.match(r'^## (\d+)\.(\d+) (.+)$', line)
        if match and current_section:
            current_subsection = {
                'type': 'subsection',
                'old_num': f"{match.group(1)}.{match.group(2)}",
                'title': match.group(3),
                'subsubsections': [],
                'content': []
            }
            current_section['subsections'].append(current_subsection)
            current_subsubsection = None
            i += 1
            continue
        
        # Sub-subsection: ### N.M.K TITLE
        match = re.match(r'^### (\d+)\.(\d+)\.(\d+) (.+)$', line)
        if match and current_subsection:
            current_subsubsection = {
                'type': 'subsubsection',
                'old_num': f"{match.group(1)}.{match.group(2)}.{match.group(3)}",
                'title': match.group(4),
                'content': []
            }
            current_subsection['subsubsections'].append(current_subsubsection)
            i += 1
            continue
        
        # Regular content line
        if current_subsubsection:
            current_subsubsection['content'].append(line)
        elif current_subsection:
            current_subsection['content'].append(line)
        elif current_section:
            current_section['content'].append(line)
        
        i += 1
    
    return sections


def renumber_sections(sections):
    """Assign new sequential numbers starting from 1."""
    for new_num, section in enumerate(sections, start=1):
        section['new_num'] = new_num
        
        for sub_idx, subsection in enumerate(section['subsections'], start=1):
            subsection['new_num'] = f"{new_num}.{sub_idx}"
            
            for subsub_idx, subsubsection in enumerate(subsection['subsubsections'], start=1):
                subsubsection['new_num'] = f"{new_num}.{sub_idx}.{subsub_idx}"


def build_output(sections):
    """Build the complete output with renumbered sections."""
    output = []
    
    for section in sections:
        # Section header with separator
        output.append('# =============================================================================')
        output.append(f"# {section['new_num']}. {section['title']}")
        output.append('# =============================================================================')
        
        # Section content (before first subsection)
        for line in section['content']:
            # Stop at first subsection
            if line.startswith('##'):
                break
            output.append(line)
        
        # Subsections
        for subsection in section['subsections']:
            output.append(f"## {subsection['new_num']} {subsection['title']}")
            
            # Subsection content (before first sub-subsection)
            for line in subsection['content']:
                if line.startswith('###'):
                    break
                output.append(line)
            
            # Sub-subsections
            for subsubsection in subsection['subsubsections']:
                output.append(f"### {subsubsection['new_num']} {subsubsection['title']}")
                
                # Sub-subsection content
                for line in subsubsection['content']:
                    output.append(line)
    
    return output


def generate_toc(sections):
    """Generate Table of Contents."""
    toc = [
        '# =============================================================================',
        '# TABLE OF CONTENTS',
        '# =============================================================================',
    ]
    
    for section in sections:
        toc.append(f"# {section['new_num']}. {section['title']}")
        
        if section['subsections']:
            for subsection in section['subsections']:
                toc.append(f"#   {subsection['new_num']} {subsection['title']}")
                
                if subsection['subsubsections']:
                    for subsubsection in subsection['subsubsections']:
                        toc.append(f"#     {subsubsection['new_num']} {subsubsection['title']}")
        
        toc.append('#')
    
    return toc


def main():
    cursorrules_path = Path(__file__).parent / '.cursorrules'
    
    if not cursorrules_path.exists():
        print(f"Error: {cursorrules_path} not found")
        return
    
    # Read file
    content = cursorrules_path.read_text()
    lines = content.splitlines()
    
    # Parse sections
    sections = parse_sections(lines)
    
    print(f"Found {len(sections)} sections:")
    for section in sections:
        subsec_count = len(section['subsections'])
        print(f"  {section['old_num']}. {section['title']} ({subsec_count} subsections)")
    
    # Renumber
    renumber_sections(sections)
    
    # Generate ToC
    toc = generate_toc(sections)
    
    # Build output
    output = [
        '# Cursor Rules for Dynamo Project',
        '# https://github.com/keivenchang/dynamo-utils/blob/main/.cursorrules',
        ''
    ]
    output.extend(toc)
    output.append('')
    output.extend(build_output(sections))
    
    # Write back
    cursorrules_path.write_text('\n'.join(output) + '\n')
    print(f"\n✅ Updated {cursorrules_path}")
    
    print("\nNew numbering:")
    for section in sections:
        print(f"  {section['new_num']}. {section['title']}")


if __name__ == '__main__':
    main()
