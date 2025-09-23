# shuffle_prompts_remove_zero_shot.py

import re
import pandas as pd
import random

# -------- CONFIG --------
INPUT_PATH  = r"C:\Users\Admin\Downloads\prompts.txt"   # tab-separated input
OUTPUT_PATH = r"C:\Users\Admin\Downloads\prompts_with_wrong_explanations.csv"
PROMPT_COL  = "prompt"          # full prompt col
ZERO_SHOT_COL = "zero_shot_prompt"  # original zero-shot col
NUM_VARIANTS = 2
RANDOM_SEED = 42
# ------------------------

random.seed(RANDOM_SEED)

# Patterns
variable_pattern = re.compile(r'\b[a-z0-9]+_[a-z0-9_]*\b')
function_pattern = re.compile(r'\b[A-Z]?[a-z]+(?:[A-Z][a-z]+)+\b')
number_pattern   = re.compile(r'\b\d+\b')
constant_pattern = re.compile(r'\b(?:None|True|False)\b')

def collect_nonoverlapping_matches(text):
    length = len(text)
    used = [False] * (length + 1)
    matches = []

    def add_matches_for(pattern, mtype):
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            if any(used[s:e]):
                continue
            for i in range(s, e):
                used[i] = True
            matches.append({'start': s, 'end': e, 'text': m.group(0), 'type': mtype})

    add_matches_for(variable_pattern, 'var')
    add_matches_for(function_pattern, 'func')
    add_matches_for(number_pattern, 'num')
    add_matches_for(constant_pattern, 'const')

    matches.sort(key=lambda x: x['start'])
    return matches

def make_shuffled_variant(text):
    matches = collect_nonoverlapping_matches(text)
    if not matches:
        return text

    occurrences_by_type = {'var': [], 'func': [], 'num': [], 'const': []}
    for m in matches:
        occurrences_by_type[m['type']].append(m['text'])

    replacements_iter = {}
    for t, items in occurrences_by_type.items():
        if items:
            shuffled = items.copy()
            random.shuffle(shuffled)
            replacements_iter[t] = iter(shuffled)
        else:
            replacements_iter[t] = None

    out_parts = []
    last_idx = 0
    for m in matches:
        out_parts.append(text[last_idx:m['start']])
        repl_iter = replacements_iter.get(m['type'])
        if repl_iter is not None:
            try:
                repl = next(repl_iter)
            except StopIteration:
                shuffled = occurrences_by_type[m['type']].copy()
                random.shuffle(shuffled)
                repl_iter = iter(shuffled)
                replacements_iter[m['type']] = repl_iter
                repl = next(repl_iter)
            out_parts.append(repl)
        else:
            out_parts.append(m['text'])
        last_idx = m['end']

    out_parts.append(text[last_idx:])
    return ''.join(out_parts)

def create_wrong_explanations(full_prompt, zero_shot, num_variants=2):
    # Remove boilerplate and zero-shot prompt
    boilerplate1 = "Respond with a Python function in one code block."
    boilerplate2 = "Think carefully and logically, explaining your answer step by step."
    
    core_text = full_prompt
    for boiler in [boilerplate1, boilerplate2, zero_shot]:
        if boiler and boiler in core_text:
            core_text = core_text.replace(boiler, "", 1).strip()

    # Generate wrong explanations
    variants = []
    for i in range(num_variants):
        variant = make_shuffled_variant(core_text)
        if i == 0:
            variants.append(f"Here is an example of a wrong explanation: {variant}")
        else:
            variants.append(f"Here is another example of a wrong explanation: {variant}")
    
    return full_prompt + " " + " ".join(variants)

# ---- Main ----
if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH, sep="\t", encoding='utf-8')

    if PROMPT_COL not in df.columns or ZERO_SHOT_COL not in df.columns:
        print(f"ERROR: required columns '{PROMPT_COL}' and '{ZERO_SHOT_COL}' not found. Available: {df.columns.tolist()}")
        raise SystemExit(1)

    df['prompt_with_wrong_explanations'] = df.apply(
        lambda row: create_wrong_explanations(row[PROMPT_COL], row[ZERO_SHOT_COL], NUM_VARIANTS),
        axis=1
    )

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print("Saved:", OUTPUT_PATH)
