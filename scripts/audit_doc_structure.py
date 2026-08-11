#!/usr/bin/env python3
import os
import re
import sys

def audit_doc_structure(sources_dir):
    func_pattern = re.compile(
        r'^\s*(?:public|open)\s+(?:static\s+|class\s+|final\s+|mutating\s+)*func\s+([A-Za-z0-9_]+)\s*(?:<[^>]+>)?\s*\((.*?)\)\s*(async\s+)?(throws\s+)?(?:->\s*(.+))?{'
    )

    total_funcs = 0
    missing_params = []
    missing_throws = []
    missing_returns = []

    for root, _, files in os.walk(sources_dir):
        for f in files:
            if f.endswith('.swift'):
                filepath = os.path.join(root, f)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                
                lines = content.splitlines()
                
                i = 0
                while i < len(lines):
                    line = lines[i]
                    full_line = line
                    j = i
                    while '{' not in full_line and j + 1 < len(lines) and ('func ' in line or j > i):
                        j += 1
                        full_line += " " + lines[j].strip()
                    
                    match = func_pattern.search(full_line)
                    if match:
                        func_name = match.group(1)
                        params_str = match.group(2).strip()
                        is_throws = match.group(4) is not None
                        return_type = match.group(5)
                        if return_type:
                            return_type = return_type.strip()
                        
                        total_funcs += 1

                        doc_lines = []
                        check_idx = i - 1
                        while check_idx >= 0:
                            prev_line = lines[check_idx].strip()
                            if prev_line.startswith('///'):
                                doc_lines.insert(0, prev_line[3:].strip())
                                check_idx -= 1
                            elif prev_line.startswith('@') or prev_line == '' or prev_line.startswith('//'):
                                check_idx -= 1
                            else:
                                break
                        
                        doc_text = "\n".join(doc_lines)
                        rel_path = os.path.relpath(filepath, sources_dir)

                        if params_str and params_str != "":
                            has_param_tag = "- Parameter" in doc_text or "- Parameters:" in doc_text
                            if not has_param_tag:
                                missing_params.append((rel_path, i + 1, func_name, params_str))

                        if is_throws:
                            has_throws_tag = "- Throws:" in doc_text
                            if not has_throws_tag:
                                missing_throws.append((rel_path, i + 1, func_name))

                        if return_type and return_type != "Void" and return_type != "()":
                            has_returns_tag = "- Returns:" in doc_text
                            if not has_returns_tag:
                                missing_returns.append((rel_path, i + 1, func_name, return_type))

                    i = j + 1

    print(f"📊 Audited {total_funcs} public functions across {sources_dir}:")
    print(f"  • Missing '- Parameter(s)': {len(missing_params)}")
    print(f"  • Missing '- Throws:': {len(missing_throws)}")
    print(f"  • Missing '- Returns:': {len(missing_returns)}")

    return missing_params, missing_throws, missing_returns

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    sources_dir = os.path.join(project_root, "Sources")
    audit_doc_structure(sources_dir)
