import glob
import os

OLD_BASE_URL = "https://docs.macrosynergy.com/"
NEW_BASE_URL = "https://macrosynergy.readthedocs.io/latest/"

HTML_PATH = "./docs/build/html"

REDIRECT_TEMPLATE = """
<!DOCTYPE html>
<html>
    <head>
        <title>Redirecting...</title>
        <meta http-equiv="refresh" content="0; url={new_url}">
    </head>
    <body>
        <h1>Redirecting...</h1>
        <p>Redirecting {old_url} to {new_url}</p>
        <p>Redirecting to <a href="{new_url}">{new_url}</a></p>
    </body>
</html>
"""


def create_html(
    new_path: str = "",
    old_path: str = "",
    new_base_url: str = NEW_BASE_URL,
    old_base_url: str = OLD_BASE_URL,
) -> str:
    new_url = new_base_url + new_path
    old_url = old_base_url + old_path
    return REDIRECT_TEMPLATE.format(new_url=new_url, old_url=old_url)


def create_redirect(
    package_file: str,
    html_path: str,
    outfile: str = None,
    overwrite: bool = False,
    new_base_url: str = NEW_BASE_URL,
    old_base_url: str = OLD_BASE_URL,
):
    process_filename = lambda x: str(x).replace(".html", "").replace(".", "/") + ".html"
    # save each file such that the new name is html_path + / + process_filename(file)

    rel_path = os.path.relpath(package_file, html_path)
    # if the path is the same, skip

    old_path = html_path + "/" + process_filename(rel_path)

    # if both paths are the same, skip
    if old_path == package_file:
        if not overwrite:
            return

    os.makedirs(os.path.dirname(old_path), exist_ok=True)
    html_content = create_html(
        new_path=rel_path,
        old_path=process_filename(rel_path),
    )
    if outfile:
        old_path = outfile
    with open(old_path, "w", encoding="utf-8") as f:
        print(f"Creating redirect for {old_path}")
        f.write(html_content)


STATIC_PAGES = [
    ("05_dev_guide.html", "contribution_guide.html"),
    ("06_definitions.html", "common_definitions.html"),
    ("02_installation.html", "getting_started.html"),
    ("01_context.html", "index.html"),
    ("03_dataquery.html", "macrosynergy.download.html"),
    # ("04_academy.html", "academy.html"),
]


def create_dirstucture(html_path: str = HTML_PATH):
    if not os.path.exists(html_path):
        raise ValueError(f"Path {html_path} does not exist")

    if not os.path.exists(html_path + "/macrosynergy"):
        os.makedirs(html_path + "/macrosynergy")

    # copy all files starting from macrosynergy. to the new folder
    package_files = glob.glob(html_path + "/*.html", recursive=True)

    for file in package_files:
        create_redirect(package_file=file, html_path=html_path)

    # create static pages
    for old_path, new_path in STATIC_PAGES:
        old_path = html_path + "/" + old_path
        os.makedirs(os.path.dirname(old_path), exist_ok=True)
        html_content = create_html(new_path=new_path, old_path=old_path)
        with open(old_path, "w", encoding="utf-8") as f:
            print(f"Creating redirect for {old_path}")
            f.write(html_content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create redirects for the documentation"
    )
    parser.add_argument(
        "--html_path",
        type=str,
        default=HTML_PATH,
        help="Path to the html documentation",
    )

    create_dirstucture()
