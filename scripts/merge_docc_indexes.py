#!/usr/bin/env python3
import os
import json
import sys

def merge_docc_indexes(tmp_dir, output_index_file, output_metadata_file=None):
    targets = [
        "SwiftDataFrame",
        "SwiftStats",
        "SwiftPreprocessing",
        "SwiftML",
        "SwiftCluster",
        "SwiftOptimize",
        "SwiftForecast",
        "SwiftNLP",
        "SwiftExplain",
        "SwiftLLM",
        "SwiftVisualization",
        "SwiftVision",
        "SwiftDatabase",
        "SwiftAgent"
    ]
    
    combined_identifiers = []
    combined_swift_nodes = []
    
    for target in targets:
        index_json_path = os.path.join(tmp_dir, target, "index", "index.json")
        if not os.path.exists(index_json_path):
            print(f"⚠️ Warning: {index_json_path} does not exist. Skipping...")
            continue
        
        with open(index_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        ids = data.get("includedArchiveIdentifiers", [])
        for i in ids:
            if i not in combined_identifiers:
                combined_identifiers.append(i)
                
        swift_nodes = data.get("interfaceLanguages", {}).get("swift", [])
        combined_swift_nodes.extend(swift_nodes)
        
    merged_data = {
        "includedArchiveIdentifiers": combined_identifiers,
        "interfaceLanguages": {
            "swift": combined_swift_nodes
        },
        "schemaVersion": {
            "major": 0,
            "minor": 1,
            "patch": 2
        }
    }
    
    os.makedirs(os.path.dirname(output_index_file), exist_ok=True)
    with open(output_index_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False)
        
    print(f"✅ Successfully merged {len(combined_swift_nodes)} root module nodes into {output_index_file}")

    if output_metadata_file:
        metadata = {
            "bundleDisplayName": "SwiftSci",
            "bundleID": "SwiftSci",
            "schemaVersion": {
                "major": 0,
                "minor": 1,
                "patch": 0
            }
        }
        with open(output_metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        print(f"✅ Updated metadata at {output_metadata_file}")

if __name__ == "__main__":
    tmp_dir = sys.argv[1] if len(sys.argv) > 1 else "./.build/docc_tmp"
    output_index = sys.argv[2] if len(sys.argv) > 2 else "./docs/index/index.json"
    output_meta = sys.argv[3] if len(sys.argv) > 3 else "./docs/metadata.json"
    merge_docc_indexes(tmp_dir, output_index, output_meta)
