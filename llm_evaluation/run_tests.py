import os
import sys
import subprocess
import re
import csv
import json
import ast
import importlib.util
import textwrap
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------
# Basic IO helpers
# -----------------------

def read_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def read_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def flatten_field(value: str) -> str:
    if value is None:
        return ""
    s = str(value)
    return s.replace("\r", " ").replace("\n", " ").replace("\t", " ")

# -----------------------
# Helper functions (as in CodePromptEval)
# -----------------------

def get_indentation_level(line):
    return len(line) - len(line.lstrip())

def get_project_path(project_name_flat):
    init_path = os.path.join(
        os.getcwd(),
        "benchmarks",
        "CoderEval",
        "CoderEval",
        "home",
        "travis",
        "builds",
        "repos",
    )
    project = None
    for root, dirs, _ in os.walk(init_path):
        for d in dirs:
            if d == project_name_flat:
                project = os.path.join(root, d)
                break
        if project:
            break
    return project

def get_task_info(task_id, tasks_file_path="benchmarks/CoderEval/CoderEval4Python.json"):
    records = read_json_file(tasks_file_path)["RECORDS"]

    project_name_flat = None
    original_file_path = None
    task_code = None
    test_name = None
    task_name = None
    test_content = None

    for task in records:
        if task["_id"] == task_id:
            project_name_flat = task["project"].replace("/", "---")
            file_path = task["file_path"].replace(".py", "_test.py")
            task_code = task["code"]
            test_name = task["test_name"]
            task_name = task["name"]
            original_file_path = file_path.replace(".py", f"_{task_id}.py")
            break

    if project_name_flat is None:
        raise ValueError(f"No task record for id {task_id}")

    project_root = get_project_path(project_name_flat)
    if project_root is None:
        raise FileNotFoundError(f"Project root not found for {project_name_flat}")

    complete_path = os.path.join(project_root, original_file_path)

    if os.path.exists(complete_path):
        with open(complete_path, "r", encoding="utf-8") as f:
            test_content = f.read()

        if test_name == "":
            test_content = test_content.replace(
                "    except:\n        isT = False",
                "    except Exception as e:\n        isT = False\n        print(\"Error while running the task: \", e)",
            )
            test_content = test_content.replace(
                "    except: \n        isT = False",
                "    except Exception as e:\n        isT = False\n        print(\"Error while running the task: \", e)",
            )
            test_content = test_content.replace(
                "    except:\n        isT=False",
                "    except Exception as e:\n        isT = False\n        print(\"Error while running the task: \", e)",
            )
            test_content = test_content.replace(
                "    except:\n        isT= False",
                "    except Exception as e:\n        isT = False\n        print(\"Error while running the task: \", e)",
            )
    else:
        test_contents = read_json_file(
            "benchmarks/CoderEval/tests/record_testcases_map_python.json"
        )
        if task_id not in test_contents:
            raise KeyError(f"No test snippet for task {task_id} in mapping")
        test_content = test_contents[task_id]

        if test_name == "":
            test_content = test_content.replace(
                'if not isT:\n        raise Exception("Result not True!!!")', ""
            )
            tail = (
                f"\n    try:\n"
                f"        assert isT == True, \"isT is not True\"\n"
                f"        print(\"Tests passed for the task: {task_name}\")\n"
                f"    except AssertionError as e:\n"
                f"        print(\"Test failed for the task:  {task_name}\", e)"
            )
            test_content = test_content + tail

    return project_name_flat, original_file_path, task_code, test_content, test_name, task_name

def get_prompt_techniques_applied(
    is_zero, is_fewshot, is_CoT, is_persona, is_package, is_signature
):
    s = ""
    if is_zero:
        s += "Zero-shot, "
    if is_fewshot:
        s += "Few-shot, "
    if is_CoT:
        s += "CoT, "
    if is_persona:
        s += "Persona, "
    if is_package:
        s += "Package, "
    if is_signature:
        s += "Signature, "
    return s[:-2]

def extract_function_name(code_str):
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None

# ---------- env helpers (pyenv + venv) ----------

def is_python_version_installed(version):
    result = subprocess.run(
        ["pyenv", "versions", "--bare"],
        capture_output=True,
        text=True,
    )
    installed = result.stdout.split()
    abstract = [".".join(v.split(".")[:2]) for v in installed]
    return version in abstract

def get_complete_python_version(python_version):
    result = subprocess.run(
        ["pyenv", "versions", "--bare"], capture_output=True, text=True
    )
    installed_versions = result.stdout.split()
    for v in installed_versions:
        if python_version in v:
            return v
    return None

def is_standard_library(module_name):
    if module_name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            return False
        module_path = spec.origin
        std_lib_path = os.path.dirname(os.__file__)
        return module_path.startswith(std_lib_path)
    except ImportError:
        return False

def check_module_needs_install(module_name):
    modules_to_ignore = [
        "re",
        "os",
        "sys",
        "subprocess",
        "urllib",
        "src",
        "collections",
        "-r",
    ]
    if module_name == "pytz" or module_name == "six":
        return True
    if module_name.replace(" ", "") == "" or module_name in modules_to_ignore:
        return False
    if is_standard_library(module_name):
        return False
    try:
        importlib.import_module(module_name)
        return False
    except ImportError:
        return True

def get_python_version(project_name_flat):
    with open("project_versions.csv", mode="r") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if row[0] == project_name_flat:
                return row[1].strip()
    return None

def install_python(python_version):
    if not is_python_version_installed(python_version):
        print(f"Installing Python {python_version}")
        if "3.6" in python_version:
            print("Applying patch for Python 3.6.15")
            subprocess.run(["pyenv", "install", "3.6.15"], check=False, shell=False)
        else:
            subprocess.run(["pyenv", "install", python_version], check=False, shell=False)

    complete = get_complete_python_version(python_version)
    print("Python installed:", complete)
    base = os.path.expanduser(f"~/.pyenv/versions/{complete}")
    python_path = os.path.join(base, f"bin/python{python_version}")
    pip_path = os.path.join(base, f"bin/pip{python_version}")
    return python_path, pip_path

def install_requirements(req_file_path, pip_path):
    with open(req_file_path, "r") as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    for pkg in packages:
        try:
            subprocess.check_call([pip_path, "install", pkg])
        except subprocess.CalledProcessError:
            print(f"Failed to install {pkg}, continuing...")

def install_general_dependencies(project_path, pip_path):
    req_files = [
        "requirements.txt",
        "test-requirements.txt",
        "requirements-dev.txt",
        "requirements.dev.txt",
        "requirements_dev.txt",
        "requirements-test.txt",
        "requirements-swh.txt",
        "requirements-development.txt",
        "test_requirements.txt",
    ]
    for req in req_files:
        fp = os.path.join(project_path, req)
        if os.path.exists(fp):
            print("Found requirements file:", req)
            try:
                install_requirements(fp, pip_path)
            except subprocess.CalledProcessError as e:
                print("Failed to install requirements:", e)

    setup_file = os.path.join(project_path, "setup.py")
    if os.path.exists(setup_file):
        try:
            subprocess.check_call([pip_path, "install", "-e", project_path])
        except subprocess.CalledProcessError as e:
            print("Failed to install local dependencies:", e)

def install_imports(task_file_path, pip_path):
    deps = set()
    with open(task_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                parts = line.split()
                if parts[0] == "import":
                    deps.add(parts[1].split(".")[0])
                elif parts[0] == "from":
                    deps.add(parts[1].split(".")[0])
    third_party = {d for d in deps if check_module_needs_install(d)}
    mapping = {
        "yaml": "pyyaml",
        "ruamel": "ruamel.yaml",
        "git": "GitPython",
        "OpenSSL": "pyOpenSSL",
        "requests": "requests==2.25.1",
        "Crypto": "pycryptodome",
        "PIL": "Pillow",
        "fs_s3fs": "fs-s3fs",
        "dateutil": "python-dateutil",
    }
    for dep in third_party:
        pkg = mapping.get(dep, dep)
        try:
            if pkg == "six":
                subprocess.check_call(
                    [pip_path, "install", "--upgrade", "setuptools<36"]
                )
                os.environ["VIRTUALENV_NO_DOWNLOAD"] = "1"
            subprocess.check_call([pip_path, "install", pkg])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {pkg}: {e}")

def initialize_venv():
    subprocess.run(["pyenv", "init", "--path"], shell=True)
    subprocess.run(["pyenv", "init", "-"], shell=True)
    subprocess.run(["pyenv", "rehash"], shell=True)

def setup_pytest(python_path, pip_path, project_root, project_name_flat):
    env = os.environ.copy()
    env["PYTHONPATH"] = python_path
    subprocess.run([pip_path, "install", "pytest"], cwd=project_root, check=False)
    subprocess.run([pip_path, "install", "-e", "."], cwd=project_root, check=False)
    if project_name_flat == "awsteiner---o2sclpy":
        subprocess.run(["brew", "install", "hdf5"], cwd=project_root, check=False)
        subprocess.run(
            [pip_path, "install", "--no-build-isolation", "--no-cache-dir", "h5py"],
            cwd=project_root,
            check=False,
        )
        subprocess.run([pip_path, "install", "h5py"], cwd=project_root, check=False)

# -----------------------
# Worker for one prediction
# -----------------------

def evaluate_single_prediction(args):
    i, row = args
    try:
        task_id = str(row["task_id"])

        comb_tuple = row["combination_id"]
        is_zero = bool(comb_tuple[0])
        is_fewshot = bool(comb_tuple[1])
        is_CoT = bool(comb_tuple[2])
        is_persona = bool(comb_tuple[3])
        is_package = bool(comb_tuple[4])
        is_signature = False

        combination = "".join(
            [
                str(int(is_zero)),
                str(int(is_fewshot)),
                str(int(is_CoT)),
                str(int(is_persona)),
                str(int(is_package)),
                str(int(is_signature)),
            ]
        )

        prompt_tech = get_prompt_techniques_applied(
            is_zero, is_fewshot, is_CoT, is_persona, is_package, is_signature
        )
        prompt = re.sub(r"\n", "", row["prompt"])

        groundtruth_code = row["groundtruth_code"]
        test_code = row["tests"]

        (
            project_name_flat,
            original_file_path,
            task_code,
            test_content,
            test_name,
            task_name,
        ) = get_task_info(task_id)

        project_root = get_project_path(project_name_flat)
        python_version = get_python_version(project_name_flat)
        if python_version is None:
            print("No python_version entry for project", project_name_flat)
            return None

        venv_dir = os.path.join(project_root, f"{project_name_flat}_env")

        if not os.path.exists(venv_dir):
            # venvs should be pre-created; if missing, create on the fly
            pyenv_python, _ = install_python(python_version)
            initialize_venv()
            print(
                f"[Worker] Creating virtual environment in Python {python_version} for {project_name_flat}..."
            )
            subprocess.run(
                [pyenv_python, "-m", "venv", f"{project_name_flat}_env"],
                cwd=project_root,
                check=True,
            )

        python_path = os.path.join(venv_dir, "bin", "python")
        pip_path = os.path.join(venv_dir, "bin", "pip")

        if not (os.path.exists(python_path) and os.path.exists(pip_path)):
            print("[Worker] Broken venv for", project_name_flat)
            return None

        # assume dependencies already installed; if needed, you can call:
        # install_general_dependencies(project_root, pip_path)

        generated_code = row["generated_code"]
        if not generated_code or generated_code == "null":
            print(
                f"Skipping task {task_id} with combination {combination} as the generated code is empty"
            )
            return {
                "comb_id": i,
                "task_id": task_id,
                "prompt_technique": flatten_field(prompt_tech),
                "prompt": flatten_field(prompt),
                "test_result": "Failed",
                "error_message": flatten_field("Incomplete code generated"),
                "groundtruth_code": flatten_field(groundtruth_code),
                "generated_code": flatten_field(generated_code),
                "lexical_distance": "",
                "test_code": flatten_field(test_code),
                "is_zero": is_zero,
                "is_fewshot": is_fewshot,
                "is_CoT": is_CoT,
                "is_persona": is_persona,
                "is_package": is_package,
                "is_signature": is_signature,
                "prompt_tech_label": prompt_tech,
            }

        file_path = original_file_path.replace(".py", f"_{combination}.py")
        file_path = os.path.join(project_root, file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            gt_name = extract_function_name(groundtruth_code)
            gen_name = extract_function_name(generated_code)

            dedented = textwrap.dedent(generated_code)
            if gt_name and gen_name and gt_name != gen_name:
                generated_task_code = dedented.replace(gen_name, gt_name)
            else:
                generated_task_code = dedented

            first_line = task_code.splitlines()[0]
            indent_level = get_indentation_level(first_line)
            generated_task_code = (
                textwrap.indent(generated_task_code, " " * indent_level)
                + "\n\n"
            )

            patched_test = test_content.replace(task_code, generated_task_code)
            f.write(patched_test)

        print(
            f"Running the task: {task_id} ({task_name}) with combination: {combination}"
        )

        if test_name == "":
            class_output = subprocess.run(
                [python_path, file_path],
                capture_output=True,
                text=True,
                cwd=project_root,
            )
            if "Error while running the task:" in class_output.stdout:
                m = re.search(
                    r"Error while running the task: (.*)",
                    class_output.stdout,
                )
                if m:
                    class_output.stderr = m.group(1)
            elif "isT is not True" in class_output.stdout:
                class_output.stderr = "isT is not True"
        else:
            print("This is a pytest file.")
            setup_pytest(
                python_path, pip_path, project_root, project_name_flat
            )
            class_output = subprocess.run(
                [python_path, "-m", "pytest", file_path],
                capture_output=True,
                text=True,
                cwd=project_root,
            )
            if class_output.returncode == 0:
                class_output.stdout += f"Tests passed for the task: {task_id}"

        print("Class output:", class_output.stdout)
        print("Class error:", class_output.stderr)

        test_result = (
            "Passed"
            if "Tests passed for the task" in class_output.stdout
            else "Failed"
        )

        return {
            "comb_id": i,
            "task_id": task_id,
            "prompt_technique": flatten_field(prompt_tech),
            "prompt": flatten_field(prompt),
            "test_result": test_result,
            "error_message": flatten_field(class_output.stderr),
            "groundtruth_code": flatten_field(groundtruth_code),
            "generated_code": flatten_field(generated_code),
            "lexical_distance": "",
            "test_code": flatten_field(test_code),
            "is_zero": is_zero,
            "is_fewshot": is_fewshot,
            "is_CoT": is_CoT,
            "is_persona": is_persona,
            "is_package": is_package,
            "is_signature": is_signature,
            "prompt_tech_label": prompt_tech,
        }

    except Exception as e:
        print("Error running the task:", row.get("task_id"), e)
        return None

# -----------------------
# Main evaluation logic (parallel)
# -----------------------

def run_evaluation_jsonl_parallel(model_output_file, evaluation_results_file, max_workers=None):
    stats_per_tech = {}
    stats_per_task = {}

    predictions = read_jsonl_file(model_output_file)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    with open(evaluation_results_file, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "comb_id",
            "task_id",
            "prompt_technique",
            "prompt",
            "test_result",
            "error_message",
            "groundtruth_code",
            "generated_code",
            "lexical_distance",
            "test_code",
            "is_zero",
            "is_fewshot",
            "is_CoT",
            "is_persona",
            "is_package",
            "is_signature",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_single_prediction, (i, row)): i
                for i, row in enumerate(predictions)
            }

            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    continue
                # write CSV row
                writer.writerow({k: result[k] for k in fieldnames})

                # aggregate stats
                if result["test_result"] == "Failed":
                    tid = result["task_id"]
                    pt = result["prompt_tech_label"]
                    stats_per_task[tid] = stats_per_task.get(tid, 0) + 1
                    stats_per_tech[pt] = stats_per_tech.get(pt, 0) + 1

    print("Failed tests per prompt technique:\n", stats_per_tech)
    print("Failed tests per task_id:\n", stats_per_task)

# -----------------------
# CLI entrypoint
# -----------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    model_results_path = os.path.join(repo_root, "results", args.model)
    model_output_file = os.path.join(
        model_results_path, "model_output", f"model_output{args.run_id}.jsonl"
    )
    evaluation_results_file = os.path.join(
        model_results_path,
        "evaluation_results",
        f"evaluation_codereval_v{args.run_id}.csv",
    )
    os.makedirs(os.path.dirname(evaluation_results_file), exist_ok=True)

    print("Model output file:", model_output_file)
    print("Evaluation results file:", evaluation_results_file)

    run_evaluation_jsonl_parallel(
        model_output_file,
        evaluation_results_file,
        max_workers=args.workers,
    )
