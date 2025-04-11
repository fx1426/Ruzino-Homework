import zipfile
import shutil
import os
import requests
from tqdm import tqdm
import argparse


def copytree_common_to_binaries(folder, target="Debug", dst=None, dry_run=False):
    root_dir = os.getcwd()
    dst_path = os.path.join(root_dir, "Binaries", target, dst or "")
    if dry_run:
        print(f"[DRY RUN] Would copy {folder} to {dst_path}")
    else:
        src_path = os.path.join(os.path.dirname(__file__), "SDK", folder)
        for root, dirs, files in os.walk(src_path):
            relative_path = os.path.relpath(root, src_path)
            dst_dir = os.path.join(dst_path, relative_path)
            os.makedirs(dst_dir, exist_ok=True)
            for file in files:
                if file.endswith(".lib"):
                    print(f"Skipping {os.path.join(root, file)}")
                    continue
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file)
                shutil.copy2(src_file, dst_file)
        print(f"Copied {folder} to {dst_path}")


def download_with_progress(url, zip_path, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] Would download from {url} to {zip_path}")
        return

    # Ensure the directory exists
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("Content-Length", 0))
    with tqdm(total=file_size, unit="B", unit_scale=True, desc=zip_path) as pbar:
        with open(zip_path, "wb") as file_handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_handle.write(chunk)
                    pbar.update(len(chunk))


def download_and_extract(url, extract_path, folder, targets, dry_run=False):
    zip_path = os.path.dirname(__file__) + "/SDK/cache/" + url.split("/")[-1]
    if os.path.exists(zip_path):
        print(f"Using cached file {zip_path}")
    else:
        if not dry_run:
            print(f"Downloading from {url}...")
        download_with_progress(url, zip_path, dry_run)

    if dry_run:
        print(f"[DRY RUN] Would extract {zip_path} to {extract_path}")
        return

    print(f"Extracting to {extract_path}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Downloaded and extracted successfully.")
        for target in targets:
            copytree_common_to_binaries(folder, target=target, dry_run=dry_run)
    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")


openusd_version = "25.02a"


def process_usd(targets, dry_run=False, keep_original_files=True, copy_only=False):
    if not copy_only:
        # First download and extract the source files
        url = "https://github.com/PixarAnimationStudios/OpenUSD/archive/refs/tags/v{}.zip".format(
            openusd_version
        )

        zip_path = os.path.join(
            os.path.dirname(__file__), "SDK", "cache", url.split("/")[-1]
        )
        if os.path.exists(zip_path):
            print(f"Using cached file {zip_path}")
        else:
            if not dry_run:
                print(f"Downloading from {url}...")
            download_with_progress(url, zip_path, dry_run)

        # Extract the downloaded zip file
        extract_path = os.path.join(
            os.path.dirname(__file__), "SDK", "OpenUSD", "source"
        )
        if keep_original_files and os.path.exists(extract_path):
            print(f"Keeping original files in {extract_path}")
        else:
            if dry_run:
                print(f"[DRY RUN] Would extract {zip_path} to {extract_path}")
            else:
                try:
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_path)
                    print(f"Downloaded and extracted successfully.")
                except Exception as e:
                    print(f"Error extracting {zip_path}: {e}")
                    return

        # Call the build script with the specified options
        build_script = os.path.join(
            extract_path,
            "OpenUSD-{}".format(openusd_version),
            "build_scripts",
            "build_usd.py",
        )

        # Check if the user has a debug python installed
        import subprocess

        try:
            subprocess.check_output(["python_d", "--version"], stderr=subprocess.STDOUT)
            has_python_d = True
        except subprocess.CalledProcessError:
            has_python_d = False
        except FileNotFoundError:
            has_python_d = False

        if has_python_d:
            use_debug_python = "--debug-python "
        else:
            use_debug_python = ""

        for target in targets:
            vulkan_support = ""
            if "VULKAN_SDK" in os.environ:
                vulkan_support = "-DPXR_ENABLE_VULKAN_SUPPORT=ON"
            elif not copy_only:
                print(
                    "Warning: VULKAN_SDK is not in the path. Highly recommend setting it for Vulkan support."
                )
            else:
                print(
                    "Error: VULKAN_SDK is not in the path. Please set it for Vulkan support before building."
                )

            build_variant_map = {
                "Debug": "debug",
                "Release": "release",
                "RelWithDebInfo": "relwithdebuginfo",
            }
            build_variant = build_variant_map.get(target, target.lower())
            if build_variant == "relwithdebuginfo":
                openvdb_args = 'OpenVDB,"-DUSE_EXPLICIT_INSTANTIATION=OFF -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBINFO="RelWithDebInfo;Release;"" '
            else:
                openvdb_args = "OpenVDB,-DUSE_EXPLICIT_INSTANTIATION=OFF "

            no_tbb_linkage = "-DCMAKE_CXX_FLAGS=-D__TBB_NO_IMPLICIT_LINKAGE=1"
            openimageio_args = f"OpenImageIO,{no_tbb_linkage} "
            build_command = f'python {build_script} --build-args USD,"-DPXR_ENABLE_GL_SUPPORT=ON {vulkan_support}" {openvdb_args}{openimageio_args}--openvdb {use_debug_python}--ptex --openimageio --opencolorio --no-examples --no-tutorials --generator Ninja --build-variant {build_variant} {os.path.dirname(__file__)}/SDK/OpenUSD/{target} -v'

            if dry_run:
                print(f"[DRY RUN] Would run: {build_command}")
            else:
                os.system(build_command)

    # Copy the built binaries to the Binaries folder
    for target in targets:
        copytree_common_to_binaries(
            os.path.join("OpenUSD", target, "bin"), target=target, dry_run=dry_run
        )
        copytree_common_to_binaries(
            os.path.join("OpenUSD", target, "lib"), target=target, dry_run=dry_run
        )
        copytree_common_to_binaries(
            os.path.join("OpenUSD", target, "plugin"), target=target, dry_run=dry_run
        )

        # Copy libraries and resources wholly
        copytree_common_to_binaries(
            os.path.join("OpenUSD", target, "libraries"),
            target=target,
            dst="libraries",
            dry_run=dry_run,
        )
        copytree_common_to_binaries(
            os.path.join("OpenUSD", target, "resources"),
            target=target,
            dst="resources",
            dry_run=dry_run,
        )


import concurrent.futures
import subprocess


def pack_sdk(dry_run=False):
    src_dir = os.path.join(os.path.dirname(__file__), "SDK")
    dst_dir = os.path.join(os.path.dirname(__file__), "SDK_temp")

    # Path that need to be replaced
    where_python = (
        subprocess.check_output(["where", "python"]).decode("utf-8").split("\n")[0]
    )
    python_dir_backward_slash = os.path.dirname(where_python).replace("/", "\\")
    python_dir_forward_slash = python_dir_backward_slash.replace("\\", "/")
    framework3d_dir_backward_slash = os.getcwd().replace("/", "\\")
    framework3d_dir_forward_slash = framework3d_dir_backward_slash.replace("\\", "/")
    vulkan_sdk_dir_backward_slash = os.environ.get("VULKAN_SDK", "").replace("/", "\\")
    vulkan_sdk_dir_forward_slash = vulkan_sdk_dir_backward_slash.replace("\\", "/")

    def copy_file(src_file, dst_file):
        if dry_run:
            print(f"[DRY RUN] Would copy {src_file} to {dst_file}")
        else:
            shutil.copy2(src_file, dst_file)
            try:
                with open(dst_file, "r", encoding="utf-8") as file:
                    filedata = file.read()
            except (UnicodeDecodeError, IOError) as e:
                return
            filedata_0 = filedata
            filedata = filedata.replace(
                python_dir_backward_slash, "{PYTHON_DIR_BACKWARD_SLASH}"
            )
            filedata = filedata.replace(
                python_dir_forward_slash, "{PYTHON_DIR_FORWARD_SLASH}"
            )
            filedata = filedata.replace(
                framework3d_dir_backward_slash, "{FRAMEWORK3D_DIR_BACKWARD_SLASH}"
            )
            filedata = filedata.replace(
                framework3d_dir_forward_slash, "{FRAMEWORK3D_DIR_FORWARD_SLASH}"
            )
            filedata = filedata.replace(
                vulkan_sdk_dir_backward_slash, "{VULKAN_SDK_DIR_BACKWARD_SLASH}"
            )
            filedata = filedata.replace(
                vulkan_sdk_dir_forward_slash, "{VULKAN_SDK_DIR_FORWARD_SLASH}"
            )
            if filedata != filedata_0:
                with open(dst_file, "w", encoding="utf-8") as file:
                    file.write(filedata)
                    print(f"Found and replaced path in {dst_file}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for root, dirs, files in os.walk(src_dir):
            # Skip build, cache directories and anything under */src/
            if any(
                skip_dir in root
                for skip_dir in ["\\build", "\\cache", "\\src", "\\source"]
            ):
                continue

            # Create corresponding directory in destination
            relative_path = os.path.relpath(root, src_dir)
            dst_path = os.path.join(dst_dir, relative_path)
            if not dry_run:
                os.makedirs(dst_path, exist_ok=True)

            for file in files:
                if file.endswith(".pdb") or file == "libopenvdb.lib":
                    print(f"Skipping {os.path.join(root, file)}")
                    continue

                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_path, file)
                futures.append(executor.submit(copy_file, src_file, dst_file))

        # Wait for all threads to complete
        concurrent.futures.wait(futures)

        # Pack the SDK_temp directory into SDK.zip
        if dry_run:
            print(f"[DRY RUN] Would pack {dst_dir} into SDK.zip")
        else:
            shutil.make_archive("SDK", "zip", dst_dir)
            print(f"Packed {dst_dir} into SDK.zip")

        # Delete the SDK_temp directory
        if dry_run:
            print(f"[DRY RUN] Would delete {dst_dir}")
        else:
            shutil.rmtree(dst_dir)
            print(f"Deleted {dst_dir}")


def find_and_replace(file_path, replacements):
    """处理单个文件的替换操作"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            filedata = file.read()

        filedata_0 = filedata
        for old_text, new_text in replacements.items():
            filedata = filedata.replace(old_text, new_text)

        if filedata != filedata_0:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(filedata)
                print(f"Found and replaced path in {file_path}")
    except (UnicodeDecodeError, IOError) as e:
        return


def main():
    parser = argparse.ArgumentParser(description="Download and configure libraries.")
    parser.add_argument(
        "--build_variant", nargs="*", default=["Debug"], help="Specify build variants."
    )
    parser.add_argument(
        "--library",
        choices=["slang", "openusd", "d3d12", "dxc"],
        help="Specify the library to configure.",
    )
    parser.add_argument("--all", action="store_true", help="Configure all libraries.")
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Print actions without executing them.",
    )
    parser.add_argument(
        "--keep-original-files",
        type=bool,
        default=True,
        help="Keep original files if the extract path exists.",
    )
    parser.add_argument(
        "--copy-only",
        action="store_true",
        help="Only copy files, skip downloading and building.",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help="Pack SDK files to SDK_temp, skipping pdb files and build/cache directories.",
    )
    args = parser.parse_args()

    targets = args.build_variant
    dry_run = args.dry_run
    keep_original_files = args.keep_original_files
    copy_only = args.copy_only

    if args.pack:
        pack_sdk(dry_run)
        return

    if args.all:
        args.library = ["openusd", "slang", "d3d12", "dxc"]
    elif not args.library:
        print(
            "No library specified and --all not set. No libraries will be configured."
        )
        return
    else:
        args.library = [args.library]

    if dry_run:
        print(f"[DRY RUN] Selected build variants: {targets}")

    if os.name == "nt":
        urls = {
            "slang": "https://github.com/shader-slang/slang/releases/download/v2025.6.3/slang-2025.6.3-windows-x86_64.zip",
            "d3d12": "https://www.nuget.org/api/v2/package/Microsoft.Direct3D.D3D12/1.613.0",
            "dxc": "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2407/dxc_2024_07_31.zip",
        }
    elif os.name == "posix":
        urls = {
            "slang": "https://github.com/shader-slang/slang/releases/download/v2024.15.2/slang-2024.15.2-macos-x86_64.zip",
            "dxc": "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2407/dxc_2024_07_31.zip",
        }
    else:
        urls = {
            "slang": "https://github.com/shader-slang/slang/releases/download/v2024.14.5/slang-2024.14.5-linux-x86_64.zip",
            "dxc": "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2407/dxc_2024_07_31.zip",
        }
    folders = {"slang": "slang/bin", "d3d12": "d3d12/bin", "dxc": "dxc/bin"}

    if copy_only and not dry_run:
        # Path that need to be replaced
        where_python = (
            subprocess.check_output(["where", "python"]).decode("utf-8").split("\n")[0]
        )
        # Check if the version of python is 3.10.11
        python_version = subprocess.check_output(
            ["python", "--version"], stderr=subprocess.STDOUT
        ).decode("utf-8")
        print(f"The highest priority Python version is {python_version}.")
        if "3.10.11" not in python_version:
            # 优先级最高的python版本应为3.10.11
            print("Please set Python version 3.10.11 as the highest priority.")
            return
        python_dir_backward_slash = os.path.dirname(where_python).replace("/", "\\")
        python_dir_forward_slash = python_dir_backward_slash.replace("\\", "/")
        framework3d_dir_backward_slash = os.getcwd().replace("/", "\\")
        framework3d_dir_forward_slash = framework3d_dir_backward_slash.replace(
            "\\", "/"
        )
        vulkan_sdk_dir_backward_slash = os.environ.get("VULKAN_SDK", "").replace(
            "/", "\\"
        )
        vulkan_sdk_dir_forward_slash = vulkan_sdk_dir_backward_slash.replace("\\", "/")

        # 创建替换映射
        replacements = {
            "{PYTHON_DIR_BACKWARD_SLASH}": python_dir_backward_slash,
            "{PYTHON_DIR_FORWARD_SLASH}": python_dir_forward_slash,
            "{FRAMEWORK3D_DIR_BACKWARD_SLASH}": framework3d_dir_backward_slash,
            "{FRAMEWORK3D_DIR_FORWARD_SLASH}": framework3d_dir_forward_slash,
            "{VULKAN_SDK_DIR_BACKWARD_SLASH}": vulkan_sdk_dir_backward_slash,
            "{VULKAN_SDK_DIR_FORWARD_SLASH}": vulkan_sdk_dir_forward_slash,
        }

        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for root, _, files in os.walk(os.path.join(os.getcwd(), "SDK")):
                for file in files:
                    file_path = os.path.join(root, file)
                    futures.append(
                        executor.submit(find_and_replace, file_path, replacements)
                    )

            # 等待所有任务完成
            concurrent.futures.wait(futures)

    for lib in args.library:
        if lib == "openusd":
            process_usd(targets, dry_run, keep_original_files, copy_only)
        elif lib == "d3d12" and os.name == "nt":
            if not copy_only:
                # Download the nupkg file
                nupkg_path = os.path.dirname(__file__) + "/SDK/cache/d3d12.nupkg"
                download_with_progress(urls[lib], nupkg_path, dry_run)

                # Rename to zip and extract
                zip_path = nupkg_path.replace(".nupkg", ".zip")

                if dry_run:
                    print(f"[DRY RUN] Would rename {nupkg_path} to {zip_path}")
                else:
                    if os.path.exists(nupkg_path):
                        shutil.copy2(nupkg_path, zip_path)
                        print(f"Renamed {nupkg_path} to {zip_path}")

                # Extract the zip file
                extract_path = os.path.dirname(__file__) + "/SDK/d3d12"
                if dry_run:
                    print(f"[DRY RUN] Would extract {zip_path} to {extract_path}")
                else:
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(extract_path)
                        print(f"Downloaded and extracted successfully.")

                        # Create bin directory and move necessary files
                        bin_dir = os.path.join(extract_path, "bin")
                        os.makedirs(bin_dir, exist_ok=True)

                        # Move relevant DLLs from extracted structure to bin folder
                        agility_path = os.path.join(
                            extract_path, "build", "native", "bin", "x64"
                        )
                        if os.path.exists(agility_path):
                            for file in os.listdir(agility_path):
                                if file.endswith(".dll") or file.endswith(".pdb"):
                                    shutil.copy2(
                                        os.path.join(agility_path, file), bin_dir
                                    )

                        print(f"D3D12 Agility SDK files prepared in {bin_dir}")
                    except Exception as e:
                        print(f"Error extracting {zip_path}: {e}")

            # Copy the D3D12 files to the binaries folder
            for target in targets:
                copytree_common_to_binaries(
                    folders[lib], target=target, dry_run=dry_run
                )
        elif lib == "dxc":
            if not copy_only:
                # Download and extract DXC
                extract_path = os.path.dirname(__file__) + "/SDK/dxc"
                zip_path = os.path.dirname(__file__) + "/SDK/cache/dxc.zip"
                download_with_progress(urls[lib], zip_path, dry_run)

                if dry_run:
                    print(f"[DRY RUN] Would extract {zip_path} to {extract_path}")
                else:
                    try:
                        # Ensure bin directory exists
                        bin_dir = os.path.join(extract_path, "bin")
                        os.makedirs(bin_dir, exist_ok=True)

                        # Extract DXC files
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(extract_path)
                        print(f"Downloaded and extracted DXC successfully.")

                        # Find and move binaries to bin directory
                        for root, _, files in os.walk(extract_path):
                            for file in files:
                                if (
                                    file.endswith(".exe")
                                    or file.endswith(".dll")
                                    or file.endswith(".lib")
                                ):
                                    src_file = os.path.join(root, file)
                                    dst_file = os.path.join(bin_dir, file)
                                    if src_file != dst_file:
                                        shutil.copy2(src_file, bin_dir)

                        print(f"DXC files prepared in {bin_dir}")
                    except Exception as e:
                        print(f"Error extracting DXC: {e}")

            # Copy the DXC files to the binaries folder
            for target in targets:
                copytree_common_to_binaries(
                    folders[lib], target=target, dry_run=dry_run
                )
        else:
            if not copy_only:
                download_and_extract(
                    urls[lib],
                    os.path.dirname(__file__) + f"/SDK/{lib}",
                    folders[lib],
                    targets,
                    dry_run,
                )
            else:
                for target in targets:
                    copytree_common_to_binaries(
                        folders[lib], target=target, dry_run=dry_run
                    )


if __name__ == "__main__":
    main()
