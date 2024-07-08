import re

def convert_requirements(input_file: str, output_file: str):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    converted_lines = []
    for line in lines:
        # Remove whitespace and newlines
        line = line.strip()
        
        # Check for the >= pattern and convert it to ==
        match = re.match(r"(.+?)\s*>=\s*([\d\.]+)", line)
        if match:
            package, version = match.groups()
            converted_lines.append(f"{package.strip()} == {version.strip()}\n")

    with open(output_file, 'w') as outfile:
        outfile.writelines(converted_lines)

# Example usage
input_file = 'requirements.txt'
output_file = 'requirements-min.txt'
convert_requirements(input_file, output_file)