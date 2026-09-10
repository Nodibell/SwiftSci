#!/usr/bin/env python3
"""
doc_tag_fixer.py — Auto-patches missing -Parameters:/-Throws:/-Returns: tags in SwiftSci docstrings.

For each public func that has /// but is missing structured DocC tags, this script:
  1. Parses the existing /// block.
  2. Appends missing tags immediately before the closing /// line.
  3. Writes the file back in-place.

Run: python3 scripts/doc_tag_fixer.py [--dry-run]
"""

import os
import re
import sys
import argparse
from dataclasses import dataclass
from typing import Optional

# Matches a public/open func declaration (possibly spanning continuation lines after the opening '(')
FUNC_DECL_RE = re.compile(
    r'^\s*(?:public|open)\s+'
    r'(?:static\s+|class\s+|final\s+|mutating\s+|nonisolated\s+)*'
    r'func\s+([A-Za-z0-9_]+)\s*'
    r'(?:<[^>]+>)?\s*\('
)

def collect_doc_block(lines: list[str], func_line_idx: int) -> tuple[int, int]:
    """Returns (start_idx, end_idx) inclusive of the /// block above func_line_idx."""
    end_idx = func_line_idx - 1
    # Skip @attributes and blank lines between doc block and func
    while end_idx >= 0 and (lines[end_idx].strip() == '' or lines[end_idx].strip().startswith('@')):
        end_idx -= 1
    if end_idx < 0 or not lines[end_idx].strip().startswith('///'):
        return (-1, -1)
    start_idx = end_idx
    while start_idx > 0 and lines[start_idx - 1].strip().startswith('///'):
        start_idx -= 1
    return (start_idx, end_idx)


def parse_params(lines: list[str], func_line_idx: int) -> list[str]:
    """
    Extracts parameter names from a func signature, handling multi-line declarations.
    Returns a list of external label (or internal name) strings.
    """
    # Gather the full signature up to the closing ')'
    raw = ''
    depth = 0
    for i in range(func_line_idx, min(func_line_idx + 8, len(lines))):
        raw += ' ' + lines[i]
        depth += lines[i].count('(') - lines[i].count(')')
        if depth <= 0:
            break
    # Extract everything between the first '(' and matching ')'
    m = re.search(r'\((.+)\)', raw, re.DOTALL)
    if not m:
        return []
    inner = m.group(1)
    # Remove generic constraints and split by comma (naively)
    params = []
    depth = 0
    current = ''
    for ch in inner:
        if ch in '(<':
            depth += 1
            current += ch
        elif ch in ')>':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            params.append(current.strip())
            current = ''
        else:
            current += ch
    if current.strip():
        params.append(current.strip())

    names = []
    for p in params:
        # Pattern: [label] name: Type  OR  name: Type
        m2 = re.match(r'^(_|[A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*:', p)
        if m2:
            label = m2.group(1)
            if label == '_':
                names.append(m2.group(2))
            else:
                names.append(label)
        else:
            m3 = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*:', p)
            if m3:
                names.append(m3.group(1))
    return names


def parse_throws_and_return(lines: list[str], func_line_idx: int) -> tuple[bool, Optional[str]]:
    """Returns (is_throwing, return_type_or_None)."""
    raw = ''
    for i in range(func_line_idx, min(func_line_idx + 8, len(lines))):
        raw += ' ' + lines[i]
        if '{' in lines[i]:
            break
    raw = ' '.join(raw.split())
    is_throws = bool(re.search(r'\bthrows\b', raw))
    m = re.search(r'->\s*(.+?)\s*(?:\{|$)', raw)
    ret = None
    if m:
        ret = m.group(1).strip()
        if ret in ('Void', '()', ''):
            ret = None
    return is_throws, ret


def get_indent(line: str) -> str:
    return re.match(r'^(\s*)', line).group(1)


def fix_file(filepath: str, dry_run: bool) -> int:
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    patched = 0
    new_lines = list(lines)
    offset = 0  # tracks line insertions

    for original_idx in range(len(lines)):
        line = lines[original_idx]
        if not FUNC_DECL_RE.search(line):
            continue

        adjusted_idx = original_idx + offset
        start, end = collect_doc_block(new_lines, adjusted_idx)
        if start == -1:
            continue  # no doc block

        doc_text = ''.join(new_lines[start:end + 1])

        has_params_tag = '- Parameter' in doc_text
        has_throws_tag = '- Throws:' in doc_text
        has_returns_tag = '- Returns:' in doc_text

        param_names = parse_params(new_lines, adjusted_idx)
        is_throws, ret_type = parse_throws_and_return(new_lines, adjusted_idx)

        tags_to_add: list[str] = []
        indent = get_indent(new_lines[adjusted_idx])

        if param_names and not has_params_tag:
            tags_to_add.append(f'{indent}/// - Parameters:\n')
            for name in param_names:
                tags_to_add.append(f'{indent}///   - {name}: <#description#>\n')

        if is_throws and not has_throws_tag:
            tags_to_add.append(f'{indent}/// - Throws: <#error description#>\n')

        if ret_type and not has_returns_tag:
            tags_to_add.append(f'{indent}/// - Returns: <#description#>\n')

        if not tags_to_add:
            continue

        # Insert after the last /// line (end index)
        insert_pos = end + 1
        new_lines[insert_pos:insert_pos] = tags_to_add
        offset += len(tags_to_add)
        patched += 1

    if not dry_run and patched > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    return patched


def main():
    parser = argparse.ArgumentParser(description='Auto-patch missing DocC tags in SwiftSci')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without writing')
    parser.add_argument('--module', default=None, help='Limit to a specific source module (e.g. SwiftStats)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sources_dir = os.path.abspath(os.path.join(script_dir, '..', 'Sources'))
    if args.module:
        sources_dir = os.path.join(sources_dir, args.module)

    total_files = 0
    total_patched = 0
    for root, _, files in os.walk(sources_dir):
        for fname in sorted(files):
            if not fname.endswith('.swift'):
                continue
            fp = os.path.join(root, fname)
            count = fix_file(fp, dry_run=args.dry_run)
            if count:
                rel = os.path.relpath(fp, sources_dir)
                print(f'  {"[DRY]" if args.dry_run else "✍️ "} {rel}: +{count} tag block(s)')
                total_patched += count
                total_files += 1

    mode = 'DRY RUN' if args.dry_run else 'PATCHED'
    print(f'\n{"🔍" if args.dry_run else "✅"} {mode}: {total_patched} tag blocks across {total_files} files.')

if __name__ == '__main__':
    main()
