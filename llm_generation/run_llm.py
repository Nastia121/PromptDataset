import re
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from model_lib import Model

load_dotenv()


def read_csv(file_path, delimiter=';'):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return pd.read_csv(file_path, delimiter=delimiter, low_memory=False)


def write_csv(df, file_path):
    df.to_csv(file_path, index=False)


def extract_code_from_response(response, task_id):
    try:
        return re.search(r'```python(.*?)```', response, re.DOTALL).group(1).strip()
    except:
        try:
            return re.search(r'```(.*?)```', response, re.DOTALL).group(1).strip()
        except:
            print(f"Could not extract code for task {task_id}")
            return ""


def process_row(myModel, item):
    """Run model and extract code for one task."""
    task_id = item['task_id']
    prompt = item['prompt']
    combination_id = item['combination_id']

    try:
        response = myModel.get_response([], prompt)
        extracted_code = extract_code_from_response(response, task_id)
    except Exception as e:
        print(f"Error for task {task_id}: {e}")
        response, extracted_code = "", ""

    return item.name, extracted_code, response


def main(args):
    dataset_filename = r"" # path to datset
    output_file = r"" # path to output destination

    data = read_csv(dataset_filename, delimiter=';')
    if data is None:
        return

    if 'generated_code' not in data:
        data['generated_code'] = ""
    if 'complete_response' not in data:
        data['complete_response'] = ""

    myModel = Model.get(
        model_name_or_path=args.model,
        max_new_tokens=1024,
        temperature=0.2
    )

    # ✅ Thread pool for parallel model calls
    max_workers = args.workers
    print(f"⚙️ Using {max_workers} parallel workers...")

    results = []
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, item in data.iterrows():
            futures.append(executor.submit(process_row, myModel, item))

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Processing")):
            idx, code, resp = future.result()
            data.at[idx, 'generated_code'] = code
            data.at[idx, 'complete_response'] = resp

            # Save progress every 50 items
            if (i + 1) % 50 == 0:
                write_csv(data, output_file)

    write_csv(data, output_file)
    print(f"✅ Done with {len(data)} tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT via OpenRouter API")
    parser.add_argument('--model', type=str, default="openai/gpt-4o", help='Model to use')
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel threads')
    args = parser.parse_args()
    main(args)
