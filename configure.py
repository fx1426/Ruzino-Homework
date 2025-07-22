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


openusd_version = "25.05.01"


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
            build_variant_map = {
                "Debug": "debug",
                "Release": "release",
                "RelWithDebInfo": "relwithdebuginfo",
            }
            build_variant = build_variant_map.get(target, target.lower())
            if build_variant == "relwithdebuginfo":
                openvdb_args = 'OpenVDB,"-DUSE_EXPLICIT_INSTANTIATION=OFF -DCMAKE_MAP_IMPORTED_CONFIG_RELWITHDEBUGINFO="RelWithDebInfo;Release;"" '
            else:
                openvdb_args = "OpenVDB,-DUSE_EXPLICIT_INSTANTIATION=OFF "

            no_tbb_linkage = "-DCMAKE_CXX_FLAGS=-D__TBB_NO_IMPLICIT_LINKAGE=1"
            openimageio_args = f"OpenImageIO,{no_tbb_linkage} "
            build_command = f'python {build_script} --build-args USD,"-DPXR_ENABLE_GL_SUPPORT=ON" {openvdb_args}{openimageio_args}--openvdb {use_debug_python}--ptex --openimageio --opencolorio --no-examples --no-tutorials --generator Ninja --build-variant {build_variant} {os.path.dirname(__file__)}/SDK/OpenUSD/{target} -v'

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
    dst_dir = os.path.join(os.path.dirname(__file__), "SDK\\SDK_pack_temp")

    # Path that need to be replaced
    where_python = (
        subprocess.check_output(["where", "python"]).decode("utf-8").split("\n")[0]
    )
    python_dir_backward_slash = os.path.dirname(where_python).replace("/", "\\")
    python_dir_forward_slash = python_dir_backward_slash.replace("\\", "/")
    framework3d_dir_backward_slash = os.getcwd().replace("/", "\\")
    framework3d_dir_forward_slash = framework3d_dir_backward_slash.replace("\\", "/")

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
                python_dir_backward_slash, "${Python3_ROOT_DIR}"
            )
            filedata = filedata.replace(
                python_dir_forward_slash, "${Python3_ROOT_DIR}"
            )
            filedata = filedata.replace(
                framework3d_dir_backward_slash, "${FRAMEWORK3D_DIR}"
            )
            filedata = filedata.replace(
                framework3d_dir_forward_slash, "${FRAMEWORK3D_DIR}"
            )
            
            # Remove brackets around paths containing placeholders
            import re
            # Pattern to match [[${FRAMEWORK3D_DIR}/...]] or [[${Python3_ROOT_DIR}/...]]
            bracket_pattern = r'\[\[(.*?)\]\]'
            matches = re.findall(bracket_pattern, filedata)
            for match in matches:
                if '${FRAMEWORK3D_DIR}' in match or '${Python3_ROOT_DIR}' in match:
                    # Normalize path separators to forward slashes
                    normalized_match = match.replace('\\', '/')
                    filedata = filedata.replace(f'[[{match}]]', normalized_match)
            
            # Also normalize any remaining paths with placeholders that have backslashes
            filedata = re.sub(r'(\$\{(?:FRAMEWORK3D_DIR|Python3_ROOT_DIR)\}[^;\s\]]*)', 
                            lambda m: m.group(1).replace('\\', '/'), filedata)

            if filedata != filedata_0:
                with open(dst_file, "w", encoding="utf-8") as file:
                    file.write(filedata)
                    print(f"Found and replaced path in {dst_file}")

    def copy_python_installation(python_dir, dst_python_dir):
        """Copy essential Python installation files"""
        if dry_run:
            print(f"[DRY RUN] Would copy Python installation from {python_dir} to {dst_python_dir}")
            return
            
        print(f"Copying Python installation from {python_dir} to {dst_python_dir}")
        os.makedirs(dst_python_dir, exist_ok=True)
        
        # Copy python.exe and python_d.exe if exists
        for exe_name in ["python.exe", "python_d.exe", "pythonw.exe"]:
            exe_path = os.path.join(python_dir, exe_name)
            if os.path.exists(exe_path):
                shutil.copy2(exe_path, dst_python_dir)
        
        # Copy all DLLs in python directory
        for file in os.listdir(python_dir):
            if file.endswith(".dll"):
                shutil.copy2(os.path.join(python_dir, file), dst_python_dir)
        
        # Copy DLLs directory if exists
        dlls_dir = os.path.join(python_dir, "DLLs")
        if os.path.exists(dlls_dir):
            dst_dlls_dir = os.path.join(dst_python_dir, "DLLs")
            shutil.copytree(dlls_dir, dst_dlls_dir, dirs_exist_ok=True)
        
        # Copy libs directory (contains python3.lib and other static libraries)
        libs_dir = os.path.join(python_dir, "libs")
        if os.path.exists(libs_dir):
            dst_libs_dir = os.path.join(dst_python_dir, "libs")
            shutil.copytree(libs_dir, dst_libs_dir, dirs_exist_ok=True)
            print(f"Copied libs directory (including python3.lib)")
        
        # Copy Scripts directory (contains pip and other tools)
        scripts_dir = os.path.join(python_dir, "Scripts")
        if os.path.exists(scripts_dir):
            dst_scripts_dir = os.path.join(dst_python_dir, "Scripts")
            shutil.copytree(scripts_dir, dst_scripts_dir, dirs_exist_ok=True)
            print(f"Copied Scripts directory (including pip)")
        
        # Copy Lib directory but exclude site-packages and other third-party packages
        lib_dir = os.path.join(python_dir, "Lib")
        if os.path.exists(lib_dir):
            dst_lib_dir = os.path.join(dst_python_dir, "Lib")
            os.makedirs(dst_lib_dir, exist_ok=True)
            
            # Standard library directories/files to include
            standard_lib_items = []
            exclude_dirs = {"site-packages", "dist-packages", "__pycache__"}
            
            for item in os.listdir(lib_dir):
                item_path = os.path.join(lib_dir, item)
                if os.path.isdir(item_path):
                    if item not in exclude_dirs:
                        standard_lib_items.append(item)
                else:
                    # Include .py files in root Lib directory
                    if item.endswith(".py"):
                        standard_lib_items.append(item)
            
            # Copy standard library items
            for item in standard_lib_items:
                src_item = os.path.join(lib_dir, item)
                dst_item = os.path.join(dst_lib_dir, item)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, dst_item)
        
        # Copy Include directory if exists
        include_dir = os.path.join(python_dir, "include")
        if os.path.exists(include_dir):
            dst_include_dir = os.path.join(dst_python_dir, "include")
            shutil.copytree(include_dir, dst_include_dir, dirs_exist_ok=True)
        
        print(f"Python installation copied successfully")

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

        # Copy Python installation
        python_dst_dir = os.path.join(dst_dir, "python")
        copy_python_installation(python_dir_backward_slash, python_dst_dir)

        # Pack the SDK_temp directory into SDK.zip
        if dry_run:
            print(f"[DRY RUN] Would pack {dst_dir} into SDK.zip")
        else:
            shutil.make_archive("SDK\\SDK", "zip", dst_dir)
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
            "slang": "https://github.com/shader-slang/slang/releases/download/v2025.12.1/slang-2025.12.1-windows-x86_64.zip",
            "d3d12": "https://globalcdn.nuget.org/packages/microsoft.direct3d.d3d12.1.616.1.nupkg",
            "dxc": "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2505.1/dxc_2025_07_14.zip",
        }
    elif os.name == "posix":
        urls = {
            "slang": "https://github.com/shader-slang/slang/releases/download/v2025.12.1/slang-2025.12.1-macos-x86_64.zip",
            "dxc": "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2505.1/dxc_2025_07_14.zip",
        }
    else:
        urls = {
            "slang": "https://github.com/shader-slang/slang/releases/download/v2025.12.1/slang-2025.12.1-linux-x86_64.zip",
            "dxc": "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.8.2505.1/dxc_2025_07_14.zip",
        }
    folders = {"slang": "slang/bin", "d3d12": "d3d12/bin", "dxc": "dxc/bin/x64"}


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
                        print(f"Error extracting DXC: {e}")            # Copy the DXC files to the binaries folder
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

    # Copy Python DLLs from SDK to Binaries for each target in copy-only mode
    if copy_only:
        sdk_python_dir = os.path.join(os.path.dirname(__file__), "SDK", "python")
        if os.path.exists(sdk_python_dir):
            for target in targets:
                bin_dir = os.path.join(os.getcwd(), "Binaries", target)
                os.makedirs(bin_dir, exist_ok=True)
                
                # Copy Python DLLs from SDK python directory
                for file in os.listdir(sdk_python_dir):
                    if file.endswith(".dll"):
                        src_file = os.path.join(sdk_python_dir, file)
                        dst_file = os.path.join(bin_dir, file)
                        if dry_run:
                            print(f"[DRY RUN] Would copy {src_file} to {dst_file}")
                        else:
                            shutil.copy2(src_file, dst_file)
                
                # Also copy DLLs from SDK python/DLLs directory if exists
                dlls_dir = os.path.join(sdk_python_dir, "DLLs")
                if os.path.exists(dlls_dir):
                    for file in os.listdir(dlls_dir):
                        if file.endswith(".dll"):
                            src_file = os.path.join(dlls_dir, file)
                            dst_file = os.path.join(bin_dir, file)
                            if dry_run:
                                print(f"[DRY RUN] Would copy {src_file} to {dst_file}")
                            else:
                                shutil.copy2(src_file, dst_file)
            print(f"Copied Python DLLs from SDK to Binaries for targets: {targets}")
        else:
            print(f"SDK Python directory not found at {sdk_python_dir}")


if __name__ == "__main__":
    main()
