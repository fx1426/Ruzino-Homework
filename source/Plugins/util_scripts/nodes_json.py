#! /usr/bin/env python
import os
import re
import json
import argparse
import concurrent.futures
from threading import Lock


def process_file(file_path, pattern, suffix="", prefix=""):
    if file_path.endswith(".cpp"):
        compiled_pattern = re.compile(pattern)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                matches = compiled_pattern.findall(content)
                if matches:
                    # Add the suffix to matches if needed
                    if len(suffix) > 0:
                        matches = list(map(lambda x: x + suffix, matches))

                    file_name_without_suffix = os.path.splitext(
                        os.path.basename(file_path)
                    )[0]

                    # Add prefix to file name if needed
                    if len(prefix) > 0:
                        file_name_without_suffix = prefix + file_name_without_suffix

                    return file_name_without_suffix, matches
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    return None


def scan_cpp_files(directories, files, pattern, suffix="", prefix=""):
    nodes = {}
    file_paths = []

    # Collect all file paths from directories
    for directory in directories:
        print(f"Scanning directory {directory}")
        for root, _, dir_files in os.walk(directory):
            for file in dir_files:
                if file.endswith(".cpp"):
                    file_paths.append(os.path.join(root, file))

    # Add individual files
    file_paths.extend([f for f in files if f.endswith(".cpp")])

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, file_path, pattern, suffix, prefix): file_path
            for file_path in file_paths
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                file_name, matches = result
                nodes[file_name] = matches

    return nodes


def main():
    parser = argparse.ArgumentParser(
        description="Scan cpp files for NODE_EXECUTION_FUNCTION and CONVERSION_EXECUTION_FUNCTION and generate JSON."
    )
    parser.add_argument(
        "--nodes-dir",
        nargs="+",
        type=str,
        help="Paths to the directories containing node cpp files",
        default=[],
    )
    parser.add_argument(
        "--nodes-files",
        nargs="+",
        type=str,
        help="Paths to the node cpp files",
        default=[],
    )
    parser.add_argument(
        "--conversions-dir",
        nargs="+",
        type=str,
        help="Paths to the directories containing conversion cpp files",
        default=[],
    )
    parser.add_argument(
        "--conversions-files",
        nargs="+",
        type=str,
        help="Paths to the conversion cpp files",
        default=[],
    )
    parser.add_argument("--username", type=str, help="Username suffix", default="")
    parser.add_argument("--output", type=str, help="Path to the output JSON file")
    args = parser.parse_args()

    result = {}
    if len(args.username) > 0:
        print("Username is specified: {}".format(args.username))
        suf = "_" + args.username
        pref = args.username + "_"
    else:
        suf = ""
        pref = ""

    if args.nodes_dir or args.nodes_files:
        node_pattern = r"NODE_EXECUTION_FUNCTION\((\w+)\)"  # match the original name
        result["nodes"] = scan_cpp_files(
            args.nodes_dir, args.nodes_files, node_pattern, suffix=suf, prefix=pref
        )
    else:
        result["nodes"] = {}

    if args.conversions_dir or args.conversions_files:
        conversion_pattern = r"CONVERSION_EXECUTION_FUNCTION\((\w+),\s*(\w+)\)"
        conversions = scan_cpp_files(
            args.conversions_dir, args.conversions_files, conversion_pattern
        )
        result["conversions"] = {
            k: [f"{match[0]}_to_{match[1]}" for match in v]
            for k, v in conversions.items()
        }
    else:
        result["conversions"] = {}

    with open(args.output, "w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=4)


if __name__ == "__main__":
    main()
