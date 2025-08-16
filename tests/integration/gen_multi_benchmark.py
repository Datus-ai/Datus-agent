import argparse
import json
import os

import yaml
from utils import fix_path


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


def generate_multi_benchmark_scripts(workdir, namespace, benchmark, instance_ids, agent_num, task_limit):
    if task_limit and task_limit < len(instance_ids):
        instance_ids = instance_ids[:task_limit]

    for agent_idx in range(1, agent_num + 1):
        output_file = f"run_integration_agent{agent_idx}.sh"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("#!/bin/bash\n\n")
            f.write("# generate test case\n")
            f.write(f"# agent: agent{agent_idx}\n")
            f.write(f"# namespace: {namespace}\n")
            f.write(f"# benchmark: {benchmark}\n")
            f.write(f"# task count: {len(instance_ids)}\n\n")

            for instance_id in instance_ids:
                command = (
                    f"(cd {workdir} && python {workdir}/datus/main.py benchmark "
                    f"--namespace {namespace} --benchmark {benchmark} "
                    f"--benchmark_task_id {instance_id} "
                    f"--config conf/multi/agent{agent_idx}.yml "
                    f"--trajectory_dir multi/agent{agent_idx}_save "
                    f"--output_dir multi/agent{agent_idx}_output)\n"
                )
                f.write(command)

        os.chmod(output_file, 0o755)
        print(f"generate to: {output_file}")

    select_script_file = "select_best_agent.sh"
    with open(select_script_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n\n")
        f.write("# select best agent script\n")
        f.write(f"# namespace: {namespace}\n")
        f.write(f"# agent_num: {agent_num}\n")
        f.write(f"# task count: {len(instance_ids)}\n\n")

        for task_id in range(len(instance_ids)):
            # TODO gold-path
            command = (
                f"python select_answer.py --workdir={workdir} --gold-path=benchmark/bird/dev_20240627/gold "
                f"--namespace {namespace} --agent={agent_num} --task-id={task_id}\n"
            )
            f.write(command)

    os.chmod(select_script_file, 0o755)
    print(f"generate to: {select_script_file}")

    print(f"generated scripts for {agent_num} agents with {len(instance_ids)} tasks each")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-agent benchmark scripts")
    parser.add_argument("--namespace", required=True, help="namespace (example: bird_sqlite)")
    parser.add_argument("--benchmark", required=True, help="benchmark (example: bird_dev)")
    parser.add_argument("--workdir", required=True, help="path to agent base directory")
    parser.add_argument("--agent_num", type=int, required=True, help="number of agents to generate scripts for")
    parser.add_argument("--task_limit", type=int, default=None, help="limit number of tasks to generate")
    parser.add_argument("--config", default="conf/agent.yml", help="agent config file")

    args = parser.parse_args()

    try:
        config_path = fix_path(args.workdir, args.config)
        config = load_config(config_path)

        benchmark_file = get_benchmark_file_path(config, args.benchmark, args.workdir)

        instance_ids = load_benchmark_data(benchmark_file)

        if not instance_ids:
            print("could not find instance_id")
            return

        generate_multi_benchmark_scripts(
            args.workdir,
            args.namespace,
            args.benchmark,
            instance_ids,
            args.agent_num,
            args.task_limit,
        )

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
