import json
import csv
import sys

def jsonl_to_csv(jsonl_path, csv_path):
    """Convert JSONL file to CSV, handling nested fields and missing keys."""
    
    # Read all JSONL lines to collect fieldnames and data
    rows = []
    all_keys = set()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                rows.append(data)
                all_keys.update(data.keys())
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num}: {e}")
                continue
    
    if not rows:
        print("No valid JSON objects found.")
        return
    
    # Sort fieldnames for consistent column order
    fieldnames = sorted(list(all_keys))
    
    # Write CSV with UTF-8 BOM for Excel compatibility
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, 
                               quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Converted {len(rows)} rows to {csv_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python jsonl_to_csv.py input.jsonl output.csv")
        sys.exit(1)
    
    jsonl_to_csv(sys.argv[1], sys.argv[2])
