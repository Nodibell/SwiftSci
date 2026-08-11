#!/usr/bin/env python3
import os
import re
import sys

def check_doc_coverage(sources_dir):
    pub_pattern = re.compile(
        r'^\s*(?:public|open)\s+(?:final\s+|actor\s+|class\s+|struct\s+|enum\s+|protocol\s+|func\s+|var\s+|let\s+|init|typealias)\b'
    )

    missing_docs = []
    total_public = 0
    documented_public = 0

    for root, _, files in os.walk(sources_dir):
        for f in files:
            if f.endswith('.swift'):
                filepath = os.path.join(root, f)
                with open(filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                for idx, line in enumerate(lines):
                    if pub_pattern.search(line):
                        if re.search(r'^\s*(?:public|open)\s+extension\b', line):
                            continue

                        total_public += 1
                        has_doc = False
                        check_idx = idx - 1
                        while check_idx >= 0:
                            prev_line = lines[check_idx].strip()
                            if prev_line.startswith('///'):
                                has_doc = True
                                break
                            elif prev_line.startswith('@') or prev_line == '' or prev_line.startswith('//'):
                                check_idx -= 1
                            else:
                                break

                        if has_doc:
                            documented_public += 1
                        else:
                            rel_path = os.path.relpath(filepath, sources_dir)
                            missing_docs.append((rel_path, idx + 1, line.strip()))

    coverage = (documented_public / total_public * 100) if total_public > 0 else 0
    print(f"📊 DocC Public API Coverage: {coverage:.2f}% ({documented_public}/{total_public} symbols documented)")

    if missing_docs:
        print(f"\n❌ FOUND {len(missing_docs)} UNDOCUMENTED PUBLIC/OPEN DECLARATIONS:")
        for path, line_num, decl in missing_docs:
            print(f"  • {path}:{line_num}: {decl}")
        print("\nCI BUILD FAILED: DocC public API coverage must be exactly 100.00%.")
        sys.exit(1)

    print("✅ 100.00% DocC Public API Coverage Verified!")
    sys.exit(0)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    sources_dir = os.path.join(project_root, "Sources")
    check_doc_coverage(sources_dir)
