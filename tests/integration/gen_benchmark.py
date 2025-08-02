import argparse
import json
import os

import yaml


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_benchmark_file_path(config, benchmark, workdir):
    benchmark_config = config.get("agent", {}).get("benchmark", {})

    if benchmark not in benchmark_config:
        raise Exception(f"benchmark '{benchmark}' not found in agent config")

    benchmark_path = benchmark_config[benchmark].get("benchmark_path")
    if not benchmark_path:
        raise Exception(f"benchmark_path not found in '{benchmark}'")

    if benchmark == "spider2":
        benchmark_file = os.path.join(workdir, benchmark_path, "spider2-snow.jsonl")
    elif benchmark == "bird_dev":
        benchmark_file = os.path.join(workdir, benchmark_path, "dev.json")
    else:
        benchmark_file = os.path.join(workdir, benchmark_path, f"{benchmark}.jsonl")

    return benchmark_file


def load_benchmark_data(benchmark_path):
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"test file not exists: {benchmark_path}")

    instance_ids = []

    # Handle JSON files (for spider2 and bird_dev)
    with open(benchmark_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

        if content.startswith("["):
            try:
                data_list = json.loads(content)
                for data in data_list:
                    if "question_id" in data:
                        instance_ids.append(data["question_id"])
                    elif "instance_id" in data:
                        instance_ids.append(data["instance_id"])
            except json.JSONDecodeError:
                print("Error: Could not parse JSON array format")
                return []
        else:
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "instance_id" in data:
                            instance_ids.append(data["instance_id"])
                        elif "question_id" in data:
                            instance_ids.append(data["question_id"])
                    except json.JSONDecodeError:
                        print(f"waring: skip error line: {line}")
                        continue

    return instance_ids


def generate_benchmark_script(workdir, namespace, benchmark, instance_ids, output_file, extra_option):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# generate test case\n")
        f.write(f"# namespace: {namespace}\n")
        f.write(f"# benchmark: {benchmark}\n")
        f.write(f"# task count: {len(instance_ids)}\n\n")

        if extra_option:
            extra_option = extra_option + " "

        for instance_id in instance_ids:
            command = (
                f"(cd {workdir} && python {workdir}/datus/main.py benchmark "
                f"--namespace {namespace} --benchmark {benchmark} "
                + extra_option
                + f"--benchmark_task_id {instance_id})\n"
            )
            f.write(command)

    os.chmod(output_file, 0o755)
    print(f"generate to: {output_file}")
    print(f"generate {len(instance_ids)} tasks")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark script")
    parser.add_argument("--namespace", required=True, help="namespace (example: snowflake)")
    parser.add_argument("--benchmark", required=True, help="benchmark (example: spider2)")
    parser.add_argument("--workdir", required=True, help="path to agent base directory")
    parser.add_argument("--output", default="run_integration.sh", help="output file")
    parser.add_argument("--config", default="conf/agent.yml", help="agent config file")
    parser.add_argument("--extra_option", default="", help="additional options to pass to script")

    args = parser.parse_args()

    try:
        config = load_config(args.workdir + "/" + args.config)

        benchmark_file = get_benchmark_file_path(config, args.benchmark, args.workdir)

        instance_ids = load_benchmark_data(benchmark_file)

        if not instance_ids:
            print("could not find instance_id")
            return

        generate_benchmark_script(
            args.workdir,
            args.namespace,
            args.benchmark,
            instance_ids,
            args.output,
            args.extra_option,
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
