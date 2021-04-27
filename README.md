![Capture](docs/source/_static/MACROSYNERGY_Logo_Primary.png)

# Macrosynergy Quant Research

Macrosynergy research package.

Contains following elements...

* management
* signal
* panel
* visual
* pnl

## Requirements
The system requires a working SQL database, with certain specifications for the schema as outlined in the [`sql/`](https://github.com/macrosynergy/qstools/tree/master/sql) folder of this repository. To clone the repository a working installation of git needs to have been installed. The program is tested in Python 3.7 and 3.8, and we recommend having upgraded pip to the latest version using
```shell script
python -m pip install --upgrade pip
```
Following this the package requirements are specified in the file  [`requirements.txt`](https://github.com/macrosynergy/macrosynergy/tree/master/requirements.txt), and should be installed using 
the command
```shell script
python -m pip install --upgrade -r requirements.txt
```

To make the documentation we need the [Sphinx package](https://www.sphinx-doc.org/),
 and  similarly for the testing framework we require the [pytest](https://docs.pytest.org/) and [flake8](https://flake8.pycqa.org/en/latest/) to have been installed:
```shell script
python -m pip install --upgrade sphinx pytest flake8
```

## Installation
To install `macrosynergy`, first clone the GitHub repository, (optionally) checkout a specific branch,  install the requirements file, and lastly install the proprietary package:
```shell script
git clone https://<username>:<password>@github.com/macrosynergy/macrosynergy
cd macrosynergy/
git checkout <branchname>
python -m pip install --upgrade -r requirements.txt
python -m pip install --upgrade ./
```
where `<username>` is your GitHub username, and `<password>` is either your GitHub password or an authentication key (required if two-factor authentication is enabled). 
The above also includes an optional step of checking out a branch different from the `master` 
in step 3: `git checkout <branchname>`. 

## Documentation
In the folder [docs/](https://github/macrosynergy/macrosynergy/tree/master/docs/) we have the documentation files created using [Sphinx](https://www.sphinx-doc.org/). To render the documentation we must use the following command (from within the `macrosynergy/` folder):
```shell script
cd docs/
make html
```
The `make html` renders the actual html files, which can be found in the folder `docs/_build/html/index.html` where the file `index.html` is the entry point to the documentation.

For a Windows machine, it might be necessary to call the `make` command using
```shell script
.\make html
```  
possibly with Administrator rights for the command line tool used.

Sphinx has three methods for being called 
1. [sphinx-build](https://www.sphinx-doc.org/en/master/man/sphinx-build.html),
2. [sphinx-autogen](https://www.sphinx-doc.org/en/master/man/sphinx-autogen.html), 
3. [sphinx-autodoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html).  

If reference API documentation for a new submodules is to be added, we can use the command (again initially from within the `macrosynergy/` folder)
```shell script
cd docs/
sphinx-apidoc -e -E -o ref/ ../macrosynergy/ --implicit-namespaces
```
where `sphinx-apidoc -o ref/ ../macrosynergy` updates the reference documentation of the [`macrosynergy`](macrosynergy/) package,
subpackage and modules.

Python's docstring conventions are described in [PEP 257 -- Docstring Conventions](https://www.python.org/dev/peps/pep-0257/) 
(see also [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
The format we are using to construct the documentations is [reSTructuredText](https://docutils.sourceforge.io/rst.html), 
which follows the [Sphinx](https://www.sphinx-doc.org/en/master/index.html) and Python standards. 
Alternative formats available that we are not using are [numpydoc docstring](https://numpydoc.readthedocs.io/en/latest/format.html) 
and [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).
For Sphinx to work we need to specify a single DOCSTRING format we are using consistently across the code base. 

## Testing and Deployment Framework
To check for any syntax errors in the code, we use `flake8` with the commands of
```shell script
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```
Further all unit and integration tests are stored in the folder [`tests/`](https://github.com/macrosynergy/macrosynergy/tree/master/tests/), and can be called using `pytest` by the command
```shell script
pytest tests/
```
We run a continuous integration and continuous deployment (CICD) framework using GitHub actions for our GitHub repository  everytime a pull request is raised into our master branch. 
The work-flows found in [`.github/workflows/`](https://github.com/macrosynergy/macrosynergy/tree/master/.github/workflows/)  and tests the package on both a Windows and a Ubuntu server. 

Similarly we will be implementing a Jenkins workflow (using a `Jenkinsfile` configuration file)  for testing and deployment on our AWS EC2 server.

### Coverage
Install the `coverage` python package using
```shell script
python -m pip install coverage
```

and run it with the following steps
```shell script
coverage run -m pytest
coverage report -m
coverage html
coverage xml
``` 
Use the `xml` to get the percent covered of the lines in testing.
Similar the coverage html report from `coverage html` can be found in the 
folder `htmlcov` and more precisely the file `htmlcov\index.html`.

