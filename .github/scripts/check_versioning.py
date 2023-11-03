import argparse
import os
import sys

sys.path.append(os.getcwd())

# get the name of the base branch
parser = argparse.ArgumentParser()

parser.add_argument(
    "--base",
    "-b",
    type=str,
    required=True,
    help="The name of the base branch",
    default="develop",
)

args = parser.parse_args()

branch_name = str(args.base).strip()


