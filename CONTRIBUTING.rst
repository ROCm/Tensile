********************************************************************
Contributing Guide
********************************************************************

Welcome to the Tensile project! If you're thinking about contributing, this document is for you. We encourage you to read this guide to understand how to contribute to the project to ensure that your contributions are accepted and merged in a timely manner.

.. tip::
   
   If you haven't already, please review :ref:`getting-started` for an introduction to the project. For details on environment setup and day-to-day development processes, please refer to the :ref:`developer-guide`.


============================
How to submit a Pull Request
============================

When making changes:

1. Create a fork of Tensile---please do not create feature branches directly in https://github.com/ROCm/Tensile.
2. Clone your fork locally and set up your :ref:`development-environment`.
3. Create a feature branch off of **develop** and make changes to the code.
4. Issue ``tox run -m precommit`` and ensure that all checks pass.
5. Commit you changes using the convention for :ref:`commit-messages`.
6. If you are updating documentation, issue ``tox run -e docs`` and verify the styling and formatting is what you expect.
7. Push the changes to your fork.

When opening a Pull Request (PR):

1. Ensure that **your develop** branch is up-to-date with the **upstream develop** branch---this may require a rebase or a merge.
2. Create the PRs against the https://github.com/ROCm/Tensile **develop** branch.
3. Fill in as many details as possible. Include a description, outcomes, notable changes, and environment information. This more information, the more likely the PR will be reviewed and merged in a timely manner.
4. Title the PR in present imperative tense, e.g., "*Update* kernel parameters" not "Updates" nor "Updated".

Once all checks pass and the PR is approved:

1. Ensure the title of the PR properly describes the changes, update if necessary.
2. Squash and merge the PR---if you are not a maintainer, a maintainer will do this for you. When merging a large change, use bullet points in the commit message to break down the changes.


After the PR is merged into the **develop** branch, it will undergo extended testing and profiling. Pending all of these checks pass, it may be promoted to staging be included in the next release. If you would like to see the changes in the next release, please ensure that the PR is merged before the release branch is cut.

------
Labels
------

.. table:: GitHub PR labels

   ============= =======
   Label         Effect
   ============= =======
   ci:profiling  Adds the *profiling* job to the CI pipeline. Profiling artifacts will be saved for 10 days.
   ci:docs-only  Only runs the *docs/readthedocs* job; omits all other pipeline jobs.
   ============= =======


===========================
Conventions and style guide
===========================

-------------------
General conventions
-------------------

1. Always use space indentation (4 spaces)---never commit a tab, e.g., ``\t``.

------------------
Python doc-strings
------------------

Tensile uses `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ to pull in documentation from doc-strings and integrate them into this site. Please use the following guidelines when writing Python functions and modules to maintain quality and consistency.

1. The all parameters and returned values should be identified with type-hints.
2. All functions should have a doc-string describing the parameters, return value, and any exception; however, if the function is small and the implementation is straightforward, a one-line doc-string is sufficient.
3. Do not include types directly in the doc-string, these should be added as type-hints in the function definition.
4. For doc-string styling, use the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.


.. _commit-messages:

---------------
Commit messages
---------------

1. Use `conventional commits <https://www.conventionalcommits.org/>`_.
2. Use the present imperative tense, e.g., "add" not "adds" nor "added".
3. Don't add a period (``.``) to the end of the message.
