#!/usr/bin/env python3
"""
Selection Results Report Generator
"""

import argparse
import glob
import json
import os
import sys

import pandas as pd


def load_selection_results(workdir):
    results = []

    search_path = os.path.join(workdir, "multi", "best_agent_output", "bird_sqlite", "selection_results_*.json")
    files = glob.glob(search_path)

    if not files:
        print(f"No selection_results files found in {search_path}")
        return results

    print(f"Found {len(files)} selection_results files")

    files.sort(key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0]))

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data and len(data) > 0:
                    results.append(data[0])
                else:
                    print(f"Warning: file {file_path} is empty or invalid")
        except Exception as e:
            print(f"Error: cannot read file {file_path}: {e}")

    return results


def analyze_results(results, num_agents):
    data = []

    for result in results:
        task_id = result.get("task_id", "Unknown")
        agent_gold_matches = result.get("agent_gold_matches", {})
        answer_found = result.get("answer_found", False)
        is_selected_agent_right = result.get("is_selected_agent_right", False)
        best_agent = result.get("best_agent", "Unknown")

        # Get instruction from agent outputs
        agent_outputs = result.get("agent_outputs", {})
        instruction = "Unknown"
        if agent_outputs:
            first_agent = list(agent_outputs.keys())[0]
            instruction = agent_outputs[first_agent].get("instruction", "Unknown")

        row = {
            "task_id": task_id,
            "answer_found": answer_found,
            "is_selected_agent_right": is_selected_agent_right,
            "best_agent": best_agent,
            "instruction": instruction,
            "raw_result": result,  # Keep raw result for detailed analysis
        }

        for i in range(1, num_agents + 1):
            agent_key = f"agent{i}"
            row[f"{agent_key}_gold_match"] = agent_gold_matches.get(agent_key, False)

        data.append(row)

    return data


def generate_report(data, num_agents, verbose=False):
    if not data:
        print("No data to analyze")
        return

    df = pd.DataFrame(data)

    print("\n" + "=" * 80)
    print("Multi-Agent Selection Results Analysis Report")
    print("=" * 80)

    if verbose:
        print("\nDetailed Results Table:")
        print("-" * 80)

        display_df = df.copy()

        # Format agent columns
        for i in range(1, num_agents + 1):
            col_name = f"agent{i}_gold_match"
            if col_name in display_df.columns:
                display_df[col_name] = display_df[col_name].map({True: "✓", False: "✗"})

        display_df["answer_found"] = display_df["answer_found"].map({True: "✓", False: "✗"})
        display_df["is_selected_agent_right"] = display_df["is_selected_agent_right"].map({True: "✓", False: "✗"})

        # Rename columns
        column_mapping = {
            "task_id": "Task ID",
            "answer_found": "Answer Found",
            "is_selected_agent_right": "Selection Correct",
            "best_agent": "Best Agent",
        }

        for i in range(1, num_agents + 1):
            column_mapping[f"agent{i}_gold_match"] = f"Agent{i} Match"

        display_df = display_df.rename(columns=column_mapping)

        print(display_df.to_string(index=False))

    total_tasks = len(df)

    print("\nOverall Statistics:")
    print("-" * 40)
    print(f"Total tasks: {total_tasks}")

    # Agent gold match statistics
    print("\nAgent Gold Match Statistics:")
    for i in range(1, num_agents + 1):
        col_name = f"agent{i}_gold_match"
        if col_name in df.columns:
            matches = df[col_name].sum()
            print(f"Agent{i} matches: {matches} ({matches/total_tasks*100:.1f}%)")

    # Answer found rate
    answer_found_count = df["answer_found"].sum()
    answer_found_rate = answer_found_count / total_tasks * 100

    print("\nAnswer Found Rate:")
    print(f"Found answers: {answer_found_count}/{total_tasks} ({answer_found_rate:.1f}%)")

    # Selection correct rate
    selected_right_count = df["is_selected_agent_right"].sum()
    selected_right_rate = selected_right_count / total_tasks * 100

    print("\nSelection Correct Rate:")
    print(f"Correct selections: {selected_right_count}/{total_tasks} ({selected_right_rate:.1f}%)")

    # Best agent distribution
    best_agent_counts = df["best_agent"].value_counts()
    print("\nBest Agent Distribution:")
    for agent, count in best_agent_counts.items():
        print(f"{agent}: {count} ({count/total_tasks*100:.1f}%)")

    # Comprehensive analysis
    print("\n" + "=" * 50)
    print("Comprehensive Analysis")
    print("=" * 50)

    # Average performance by agent
    print("\nAverage Performance by Agent:")
    for i in range(1, num_agents + 1):
        agent_key = f"agent{i}"
        col_name = f"{agent_key}_gold_match"
        if col_name in df.columns:
            match_rate = df[col_name].mean() * 100
            selected_count = (df["best_agent"] == agent_key).sum()
            selected_rate = selected_count / total_tasks * 100
            print(
                f"{agent_key}: Gold match rate {match_rate:.1f}%, "
                f"Selected as best {selected_count} times ({selected_rate:.1f}%)"
            )

    # Answer found but wrong agent selection analysis
    answer_found_wrong_selection = df[df["answer_found"] & ~df["is_selected_agent_right"]]
    wrong_selection_count = len(answer_found_wrong_selection)
    wrong_selection_rate = wrong_selection_count / answer_found_count * 100 if answer_found_count > 0 else 0

    print("\nAnswer Found but Wrong Agent Selection Analysis:")
    print("=" * 70)
    print(
        "Cases where answer was found but wrong agent was selected: "
        f"{wrong_selection_count}/{answer_found_count} ({wrong_selection_rate:.1f}%)"
    )

    if wrong_selection_count > 0:
        # Analyze patterns in wrong selections
        wrong_selection_by_agent = answer_found_wrong_selection["best_agent"].value_counts()
        print("\nWrong selections by agent:")
        for agent, count in wrong_selection_by_agent.items():
            print(f"  {agent}: {count} times ({count/wrong_selection_count*100:.1f}%)")

        if verbose:
            # Analyze error types
            selected_agent_has_gold_match = 0
            selected_agent_no_gold_match = 0

            for _, row in answer_found_wrong_selection.iterrows():
                best_agent = row["best_agent"]
                agent_num = int(best_agent.replace("agent", ""))
                col_name = f"agent{agent_num}_gold_match"
                if col_name in row and row[col_name]:
                    selected_agent_has_gold_match += 1
                else:
                    selected_agent_no_gold_match += 1

            print("\nError analysis:")
            print(
                f"  Selected agent has gold match but not optimal: {selected_agent_has_gold_match}"
                f" ({selected_agent_has_gold_match/wrong_selection_count*100:.1f}%)"
            )
            print(
                f"  Selected agent has no gold match: {selected_agent_no_gold_match}"
                f" ({selected_agent_no_gold_match/wrong_selection_count*100:.1f}%)"
            )

    if wrong_selection_count > 0 and verbose:
        print("\nDetailed Error Cases:")
        print("-" * 80)

        for _, row in answer_found_wrong_selection.iterrows():
            task_id = row["task_id"]
            best_agent = row["best_agent"]
            instruction = row["instruction"]

            # Get agent gold matches for this case
            agent_matches = []
            correct_agents = []
            for i in range(1, num_agents + 1):
                agent_key = f"agent{i}"
                col_name = f"{agent_key}_gold_match"
                if col_name in row:
                    match_status = "✓" if row[col_name] else "✗"
                    agent_matches.append(f"{agent_key}:{match_status}")
                    if row[col_name]:
                        correct_agents.append(agent_key)

            agent_matches_str = ", ".join(agent_matches)
            correct_agents_str = ", ".join(correct_agents) if correct_agents else "None"

            print(f"\nTask {task_id}:")
            print(f"  Instruction: {instruction[:100]}{'...' if len(instruction) > 100 else ''}")
            print(f"  Selected: {best_agent} (WRONG)")
            print(f"  Gold matches: {agent_matches_str}")
            print(f"  Should have selected: {correct_agents_str}")

            # Analysis of why wrong selection might have occurred
            raw_result = row["raw_result"]
            reason = raw_result.get("reason", "No reason provided")
            print(f"  Selection reason: {reason[:150]}{'...' if len(reason) > 150 else ''}")
            print("-" * 80)

    # Key metrics summary
    print("\nKey Metrics Summary:")
    print(f"• Answer found accuracy: {answer_found_rate:.1f}%")
    print(f"• Agent selection accuracy: {selected_right_rate:.1f}%")
    print(f"• Wrong agent selection rate (among found answers): {wrong_selection_rate:.1f}%")
    print(f"• Overall system success rate: {min(answer_found_rate, selected_right_rate):.1f}%")

    if not verbose and wrong_selection_count > 0:
        print("\nNote: Use --verbose flag to see detailed error cases and analysis")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-agent selection results")
    parser.add_argument("--workdir", required=True, help="Working directory path")
    parser.add_argument("--agent", type=int, default=3, help="Number of agents (default: 3)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed analysis including error cases")

    args = parser.parse_args()

    if not os.path.exists(args.workdir):
        print(f"Error: working directory {args.workdir} does not exist")
        sys.exit(1)

    if args.agent < 1:
        print("Error: number of agents must be at least 1")
        sys.exit(1)

    results = load_selection_results(args.workdir)

    if not results:
        print("No valid selection_results files found")
        sys.exit(1)

    data = analyze_results(results, args.agent)
    generate_report(data, args.agent, args.verbose)


if __name__ == "__main__":
    main()
