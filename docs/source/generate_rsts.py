import os
import inspect
import importlib
import pkgutil
import os


def create_rst_files_for_module(module, rst_dir, module_name):
    members = inspect.getmembers(module)
    for name, member in members:
        if inspect.isfunction(member) or inspect.isclass(member):
            if member.__module__ == module_name:
                # Create a file for each class or function
                filename = os.path.join(rst_dir, f"{module_name}.{name}.rst")
                with open(filename, "w") as f:
                    f.write(f'{name}\n{"=" * len(name)}\n\n')
                    f.write(f".. automodule:: {module_name}\n")
                    f.write(f"   :members: {name}\n")


def create_rst_files(package_name, rst_dir):
    package = importlib.import_module(package_name)
    os.makedirs(rst_dir, exist_ok=True)

    for importer, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        module = importlib.import_module(modname)
        create_rst_files_for_module(module, rst_dir, modname)


def generate_modules_rst(rst_dir, output_file):
    modules = set()

    # Scanning for unique module names in the .rst files
    for filename in os.listdir(rst_dir):
        if filename.endswith(".rst"):
            # Removing the '.rst' and keeping the full module path
            module_name = filename.rsplit(".", 1)[0]
            modules.add(module_name)

    with open(output_file, "w") as f:
        f.write("Modules\n")
        f.write("=======\n\n")
        f.write("Here is a list of all modules:\n\n")
        f.write(".. toctree::\n")
        f.write("   :maxdepth: 2\n\n")

        for module in sorted(modules):
            f.write(f"   {module}\n")


if __name__ == "__main__":
    create_rst_files("macrosynergy", "docs/source/rstx")
    generate_modules_rst("docs/source/rstx", "docs/source/modules.rst")
