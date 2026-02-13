import csv
import json
import re
from pathlib import Path

# allow large code blocks
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB per field

# normalize headers for mapping
def normalize_header(h):
    if h is None:
        return ""
    s = str(h).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

# parse combination_id
def parse_combination_tuple(value):
    if not value:
        flags = []
    else:
        flags = re.findall(r"(True|False)", str(value))
        flags = [f == "True" for f in flags]
    while len(flags) < 5:
        flags.append(False)
    return {
        "is_zero_shot": flags[0],
        "is_few_shot": flags[1],
        "is_chain_of_thought": flags[2],
        "is_contrastive_chain_of_thought": flags[3],
        "is_program_of_thought": flags[4],
    }

# map CSV row to canonical object
def map_row(row):
    row_norm = {normalize_header(k): v for k, v in row.items()}
    # add combination flags
    comb = row_norm.get("combination_id", "")
    row_norm.update(parse_combination_tuple(comb))
    # convert prompt length
    try:
        row_norm["original_prompt_length"] = int(row_norm.get("original_prompt_length", 0))
    except:
        row_norm["original_prompt_length"] = 0
    return row_norm

def convert(input_csv, output_jsonl):
    input_path = Path(input_csv)
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8", newline="") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in, delimiter=";", quotechar='"')
        written = 0
        for row in reader:
            if not any(v.strip() for v in row.values() if v):
                continue
            obj = map_row(row)
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    print(f"Done. Rows written: {written}. JSONL saved to {output_jsonl}")

# ------------------- CLI -------------------
if __name__ == "__main__":
    input_csv = r"results\gpt4o\model_output\model_output1_failed_prompt.csv"  
    output_jsonl = r"results\gpt4o\model_output\model_output4.jsonl"

    convert(input_csv, output_jsonl)
