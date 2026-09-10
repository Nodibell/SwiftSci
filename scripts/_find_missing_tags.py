#!/usr/bin/env python3
"""Print the exact file/line of missing DocC tags."""
import os, re, sys

FUNC_DECL_RE = re.compile(
    r'^\s*(?:public|open)\s+'
    r'(?:static\s+|class\s+|final\s+|mutating\s+|nonisolated\s+)*'
    r'func\s+([A-Za-z0-9_]+)\s*'
    r'(?:<[^>]+>)?\s*\('
)

def collect_doc(lines, idx):
    end = idx - 1
    while end >= 0 and (lines[end].strip() == '' or lines[end].strip().startswith('@')):
        end -= 1
    if end < 0 or not lines[end].strip().startswith('///'):
        return ''
    start = end
    while start > 0 and lines[start-1].strip().startswith('///'):
        start -= 1
    return ''.join(lines[start:end+1])

def parse_sig(lines, idx):
    raw = ''
    for i in range(idx, min(idx+8, len(lines))):
        raw += ' ' + lines[i]
        if '{' in lines[i]:
            break
    raw = ' '.join(raw.split())
    has_params = bool(re.search(r'\(.*?\S.*?\)', raw))
    is_throws = bool(re.search(r'\bthrows\b', raw))
    m = re.search(r'->\s*(.+?)\s*(?:\{|$)', raw)
    ret = m.group(1).strip() if m else None
    if ret in ('Void', '()', '', None):
        ret = None
    return has_params, is_throws, ret

script_dir = os.path.dirname(os.path.abspath(__file__))
sources_dir = os.path.abspath(os.path.join(script_dir, '..', 'Sources'))
for root, _, files in os.walk(sources_dir):
    for f in sorted(files):
        if not f.endswith('.swift'): continue
        fp = os.path.join(root, f)
        lines = open(fp, encoding='utf-8').readlines()
        for idx, line in enumerate(lines):
            if not FUNC_DECL_RE.search(line): continue
            doc = collect_doc(lines, idx)
            if not doc: continue
            has_params, is_throws, ret = parse_sig(lines, idx)
            rel = os.path.relpath(fp, sources_dir)
            if has_params and '- Parameter' not in doc:
                print(f"MISSING -Parameters: {rel}:{idx+1}  {line.strip()[:80]}")
            if is_throws and '- Throws:' not in doc:
                print(f"MISSING -Throws:     {rel}:{idx+1}  {line.strip()[:80]}")
