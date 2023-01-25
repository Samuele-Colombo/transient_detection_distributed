# Copyright 2022 Samuele Colombo.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import os.path as osp
import tempfile
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog = 'install_requirements.py',
        description = "Using PIP, install all packages from 'requirements.txt'",
        prefix_chars="§"
    )
    parser.add_argument("options", action="store", 
        nargs="*", default=[],
        help="Add here any `pip install` options (e.g. if you want to make a dry run you may "+
            "add `--dry-run`as argument. You must not change the `-r | --requirements` option."
    )
    parsed = parser.parse_args(sys.argv[1:])

    options = parsed.options

    if '-r' in options or "--requirements" in options:
        raise Exception("Error: you must not modify the `-r | --requirements` option.")

    requirementsfile = osp.join(os.getcwd(), "requirements.txt")

    # List of packages to exclude
    excluded_packages = ["pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric"]

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

    # Open the requirements file
    with open(requirementsfile, 'r') as f:
        # Iterate through each line in the file
        for line in f:
            # Check if the line contains one of the excluded packages
            if any(package in line for package in excluded_packages):
                # Skip this line
                continue
            # Write the line to the temporary file
            temp_file.write(line)

    # Close the file
    temp_file.close()

    # Install the packages listed in the temporary file
    subprocess.run(['pip', 'install', *options,  '-r', temp_file.name])

    # Delete the temporary file
    os.unlink(temp_file.name)

    import torch
    import requests

    this_version = torch.__version__
    pytorch_major_version, pytorch_minor_version, pytorch_patch_version = \
        this_version.split("+")[0].split(".")
    try:
        pytorch_cuda_version = this_version.split("+")[1]
    except IndexError:
        pytorch_cuda_version = "cpu"
        this_version = "+".join([this_version, pytorch_cuda_version])

    while int(pytorch_patch_version) >= 0:
        pygwheel_url = "https://data.pyg.org/whl/torch-{}.html".format(this_version)
        r = requests.get(pygwheel_url)
        if r.status_code == 200:
            # The webpage exists, so we can exit the loop
            print(f"- Installing `pytorch_geometric` packages from wheel for version {this_version}.")
            break
        else:
            # The webpage does not exist, so we reduce the patch version of PyTorch and try again
            pytorch_patch_version = str(int(pytorch_patch_version) - 1)
            print(f"- Failed to recover a wheel for `pytorch_geometric` for version {this_version}, trying for previous patch versions...", file=sys.stderr)
            this_version = "{major}.{minor}.{patch}+{cuda}".format(
                major = pytorch_major_version,
                minor = pytorch_minor_version,
                patch = pytorch_patch_version,
                cuda  = pytorch_cuda_version
            )
    else:
        raise Exception(f"Error: could not find a suitable wheel for torch version {torch__version__}")


    pygwheel = "https://data.pyg.org/whl/torch-{}.html".format(this_version)

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *options,
    *excluded_packages, "-f", pygwheel])

if __name__ == "__main__":
    main()