import re

def convert_requirements(input_file: str, output_file: str):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    converted_lines = []
    for line in lines:
        line = line.strip()
        
        match = re.match(r"(.+?)\s*>=\s*([\d\.]+)", line)
        if match:
            package, version = match.groups()
            converted_lines.append(f"{package.strip()} == {version.strip()}\n")

    with open(output_file, 'w') as outfile:
        outfile.writelines(converted_lines)

convert_requirements('requirements.txt', 'requirements-min.txt')