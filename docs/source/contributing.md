# Contributing
We welcome you to [check the existing issues](https://github.com/UrbsLab/STREAMLINE/issues/) for bugs or enhancements to work on. 
If you have an idea for an extension to STREAMLINE, please [file a new issue](https://github.com/UrbsLab/STREAMLINE/issues//new), 
or email harsh.bandhey@cshs.org to discuss it.

***
## Project Layout
The latest release of STREAMLINE is on the [main branch](https://github.com/UrbsLab/STREAMLINE/tree/main), 
whereas the legacy/Beta 0.2.5 version of STREAMLINE is on the [legacy branch](https://github.com/STREAMLINE/tree/legacy).

The in-development code is stored in the  [development branch](https://github.com/STREAMLINE/tree/dev)
Make sure you are looking at and working on the correct branch if you're looking to contribute code.

In terms of directory structure:

* All of STREAMLINE's code sources are in the `streamline` directory
* The documentation sources are in the `docs/source` directory
* Unit tests for STREAMLINE are in the `streamline/tests` module

Make sure to familiarize yourself with the project layout before making any major contributions.

***
## How to Contribute
The preferred way to contribute to STREAMLINE is to fork the 
[main repository](https://github.com/UrbsLab/STREAMLINE/) on
GitHub:

1. Fork the [project repository](https://github.com/UrbsLab/STREAMLINE/):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

          $ git clone git@github.com:YourLogin/STREAMLINE.git
          $ cd STREAMLINE

3. Create a branch to hold your changes:

          $ git checkout -b my-contribution

4. Make sure your local environment is set up correctly for development. Installation instructions are almost identical to [the user instructions](install.md)

          $ pip install -r requirements

5. Start making changes on your newly created branch, remembering to never work on the ``main`` branch! Work on this copy on your computer using Git to do the version control.

6. Once some changes are saved locally, you can use your tweaked version of STREAMLINE by navigating to the project's base directory and running STREAMLINE in a script. 

7. To check your changes haven't broken any existing tests and to check new tests you've added pass run the following (note, you must have the `pytest` package installed within your dev environment for this to work):

          $ pytest --log-cli-level=INFO

8. When you're done editing and local testing, run:

          $ git add <modified_files>
          $ git commit -m <git messages>

   to record your changes in Git, then push them to GitHub with:

          $ git push -u origin my-contribution

Finally, go to the web page of your fork of the STREAMLINE repo, and click 'Pull Request' (PR) to send your changes to the maintainers for review. Make sure that you send your PR to the `dev` branch, as the `main` branch is reserved for the latest stable release. This will start the CI server to check all the project's unit tests run and send an email to the maintainers.

(For details on the above look up the [Git documentation](http://git-scm.com/documentation) on the web.)

***
## Before Submitting a Pull Request
Before you submit a pull request for your contribution, please work through this checklist to make sure that you have done everything necessary so we can efficiently review and accept your changes.

If your contribution changes STREAMLINE in any way:

* Update the [documentation](https://github.com/UrbsLab/STREAMLINE/tree/main/docs/source) so all of your changes are reflected there.

* Update the [README](https://github.com/UrbsLab/STREAMLINE/blob/main/README.md) if anything there has changed.

If your contribution involves any code changes:

* Update the [project unit tests](https://github.com/UrbsLab/STREAMLINE/tree/main/streamine/tests) to test your code changes.

* Make sure that your code is properly commented with [docstrings](https://www.python.org/dev/peps/pep-0257/) and comments explaining your rationale behind non-obvious coding practices.

If your contribution requires a new library dependency:

* Double-check that the new dependency is easy to install via `pip`. 
* Add it to the `requirements.txt` file.

***
## Updating the Documentation
We use [sphinx](https://www.sphinx-doc.org/) to manage our [documentation](https://urbslab.github.io/STREAMLINE/). 
This allows us to write the docs in Markdown and compile them to HTML as needed. 
Below are a few useful tips/commands to know when updating the documentation.

* Install additional documentation packages to generate documentation locally 
      
      $ pip install sphinx sphinx_rtd_theme myst-parser

* Edit/Add markdown or reST files in the `docs/source` folder.
* Each new added markdown or reST file needs to be added into the `toctree` in `docs/source/index.rst` file.

* You can use the following command to creates a fresh build of the documentation in HTML. Always run this before deploying the documentation to GitHub. 

       $ sphinx-build -b html docs/source docs/build/html -E -a

***
## After Submitting a Pull Request
After submitting your pull request, GitHub Actions will automatically run unit tests on your changes and make sure that your updated code runs.

Check back shortly after submitting your pull request to make sure that your code passes these checks. 
If any of the checks come back with a red X, then do your best to address the errors.