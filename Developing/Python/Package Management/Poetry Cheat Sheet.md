#### Create a new project

```bash
poetry new <project-name>
```

#### Add a new lib

```bash
poetry add <library>
```

#### Remove a lib

```bash
poetry remove <library>
```

#### Update a lib

```bash
poetry update <library>
```

#### Get venv path

```bash
poetry run which python
```

#### Run app

```bash
poetry run python app.py
```

#### Run tests

```bash
poetry run python -m unittest discover
```

#### Show dependencies

```bash
poetry show
```

#### Create script

1 - Edit `pyproject.toml`:

```toml
[tool.poetry.scripts]
test = 'scripts:test'
```

2 - Create a `scripts.py` file on the root directory of your project:

```python
import subprocess

def test():
    """
    Run all unittests. Equivalent to:
    `poetry run python -u -m unittest discover`
    """
    subprocess.run(
        ['python', '-u', '-m', 'unittest', 'discover']
    )
```

3 - Run script:

```bash
poetry run test
```

#### Disable virtual environment creation

```bash
poetry config virtualenvs.create false
```

#### List configuratiom

```bash
poetry config --list
```

How do I deactivate a poetry environment?

To deactivate a Poetry environment, you can use one of the following methods¹:

1. **Create a nested shell with `poetry shell`**, then type `exit` to deactivate the virtual environment and exit the shell¹.
2. **Use `deactivate`** to deactivate the virtual environment without leaving the shell¹.
3. **Run `poetry env remove <environment>`** to delete the virtual environment¹.
4. **Run `rm -rf <path-to-environment>`** to remove the virtual environment directory¹.

Please note that you need to replace `<environment>` with the name of your environment and `<path-to-environment>` with the path to your environment¹.

Source: Conversation with Bing, 11/18/2023
(1) Basic usage | Documentation | Poetry - Python dependency management and .... https://python-poetry.org/docs/basic-usage/.
(2) . https://bing.com/search?q=how+to+deactivate+a+poetry+environment.
(3) Managing environments | Documentation | Poetry - Python dependency .... https://python-poetry.org/docs/managing-environments/.
(4) I can't delete an poetry managed environment - Stack Overflow. https://stackoverflow.com/questions/73121344/i-cant-delete-an-poetry-managed-environment.
(5) undefined. https://stackoverflow.com/questions/60580332/poetry-virtual-environment-already-activated.
(6) undefined. https://stackoverflow.com/questions/70064449/how-to-force-reinstall-poetry-environment.

I am using a project from someone's git repo where they used Poetry to install Python packages. I see they have a tool.poetry in a pyproject.toml. How am I supposed to use this with Poetry exactly?

The `pyproject.toml` file is a configuration file that contains all the information about your project. The `tool.poetry` section of the `pyproject.toml` file is composed of multiple sections such as the name of the package, version, description, license, authors, maintainers, readme, homepage, repository, documentation, keywords, classifiers, etc¹.

To use this project with Poetry, you would typically follow these steps:

1. **Clone the repository**: Clone the git repository to your local machine.

2. **Install Poetry**: If you haven't installed Poetry, you can do so by following the instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

3. **Navigate to the project directory**: Use the command line to navigate into the directory that contains the `pyproject.toml` file.

4. **Install dependencies**: Run the command `poetry install` in your command line. This will read the `pyproject.toml` file and install all the dependencies specified under `tool.poetry.dependencies` and `tool.poetry.dev-dependencies`¹⁴.

5. **Run the project**: You can now run the project. If there are any specific scripts defined in `tool.poetry.scripts`, you can run them using the command `poetry run <script-name>`².

Remember, any command you run with `poetry run` will execute within the virtual environment that Poetry has created for your project². This ensures that you are using the exact versions of the packages that the project requires⁴.

Source: Conversation with Bing, 11/18/2023
(1) The pyproject.toml file | Documentation | Poetry - Python dependency .... https://python-poetry.org/docs/pyproject/.
(2) Dependency Management With Python Poetry – Real Python. https://realpython.com/dependency-management-python-poetry/.
(3) how to run a script using pyproject.toml settings and poetry?. https://stackoverflow.com/questions/59286983/how-to-run-a-script-using-pyproject-toml-settings-and-poetry.
(4) python - What is pyproject.toml file for? - Stack Overflow. https://stackoverflow.com/questions/62983756/what-is-pyproject-toml-file-for.
(5) undefined. https://python-poetry.org/.
(6) undefined. https://github.com/python-poetry/poetry.
(7) undefined. https://python-poetry.org/docs/.
(8) undefined. http://192.168.1.xxx/home/whl.

How do I use Python Poetry to change environments?

To change environments with Poetry, you can use the `env use` command to tell Poetry which Python version to use for the current project¹. Here's how you can do it:

1. **Specify the Python version**: Use the command `poetry env use X.Y`, where `X.Y` is the Python version you want to use¹. For example, if you want to use Python 3.7, you would use the command `poetry env use 3.7`.

2. **Specify the Python executable**: If you have the Python executable in your PATH, you can use it directly¹. For example, you can use the command `poetry env use python3.7`¹.

3. **Specify the full path to Python**: You can also specify the full path to the Python executable¹. For example, you can use the command `poetry env use /full/path/to/python`¹.

If you want to disable the explicitly activated virtual environment and retrieve the default behavior, you can use the special system Python version³. You can do this by using the command `poetry env use system`³.

Remember, any command you run with `poetry run` will execute within the virtual environment that Poetry has created for your project³. This ensures that you are using the exact versions of the packages that the project requires³.

Source: Conversation with Bing, 11/18/2023
(1) Configure a Poetry environment | PyCharm Documentation - JetBrains. https://www.jetbrains.com/help/pycharm/poetry.html.
(2) Managing environments | Documentation | Poetry - Python dependency .... https://python-poetry.org/docs/managing-environments/.
(3) . https://bing.com/search?q=how+to+change+environments+with+Python+Poetry.
(4) Configuration | Documentation | Poetry - Python dependency management .... https://python-poetry.org/docs/configuration/.
(5) Simplify Python Dependency Management: Creating and Using Virtual .... https://dev.to/rainleander/simplify-python-dependency-management-creating-and-using-virtual-environments-with-poetry-22ee.
(6) python - How to create a brand new virtual environment or duplicate an .... https://stackoverflow.com/questions/70739858/how-to-create-a-brand-new-virtual-environment-or-duplicate-an-existing-one-in-po.
(7) undefined. https://www.jetbrains.com/help/idea/poetry.html.

This repository has a pyproject.toml file for each directory. Each directory has a lesson embedded to teach a certain part of Semantic kernel. How might I switch from lesson to lesson and activate the environment for the config file? I don't wish to reinstall the packages over and over since they are generally the same.

If the packages are generally the same for each lesson, you can create a single virtual environment for all lessons. Here's how you can do it:

1. **Navigate to the root directory**: Use the command line to navigate to the root directory that contains all the lesson directories.

2. **Create a virtual environment**: Run the command `poetry install` in your command line. This will read the `pyproject.toml` file in the root directory and install all the dependencies specified under `tool.poetry.dependencies` and `tool.poetry.dev-dependencies`.

3. **Switch between lessons**: To switch between lessons, you don't need to do anything special. Just navigate to the directory of the lesson you want to work on and start your work. Since all lessons use the same virtual environment, you don't need to activate or deactivate it when switching between lessons.

Remember, any command you run with `poetry run` will execute within the virtual environment that Poetry has created for your project. This ensures that you are using the exact versions of the packages that the project requires.

If a lesson requires a package that is not installed in the virtual environment, you can add it with the command `poetry add <package-name>`. If you no longer need a package, you can remove it with the command `poetry remove <package-name>`.

Please note that this approach assumes that the lessons are not dependent on each other and can be run independently. If a lesson depends on the state of a previous lesson, you might need to run the lessons in a specific order.

Is a pyproject.toml file the same as a poetry.lock file?

The `pyproject.toml` and `poetry.lock` files serve different purposes in a Poetry project.

The `pyproject.toml` file is a configuration file that contains all the information about your project. It includes the project's dependencies and other metadata.

On the other hand, the `poetry.lock` file is automatically generated by Poetry when you add or install dependencies. It contains the exact versions of the dependencies used in your project. This ensures that everyone working on the project uses the same versions of the dependencies.

You cannot use a `pyproject.toml` file as a lock file. If you want to generate a `poetry.lock` file, you can use the `poetry lock` command. This will create a `poetry.lock` file based on the dependencies specified in your `pyproject.toml` file.
