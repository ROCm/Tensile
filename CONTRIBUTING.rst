********************************************************************
Contribution guidelines
********************************************************************

This document provides the guidelines for contributing to the Tensile source code.

.. seealso::

   For information about environment setup and development processes, see :ref:`programmers-guide`.

Tensile's development practice is based on the `Gitflow workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. The **develop** branch is the default branch for development, where all new features and bug fixes are merged. After a PR is merged into **develop**, it undergoes extended testing and profiling. If these checks pass, the PR might be merged into **staging** to be included in the next release. A PR is available in the upcoming release only if it is merged before the release branch is cut.

============================
Submitting a Pull Request
============================

a. **Forking the repository:**

   1. Create a fork of Tensile. Do not create feature branches directly in https://github.com/ROCm/Tensile.
   2. Clone your fork locally and set up your :ref:`development-environment`.
   3. Create your feature branch from **develop** and make changes to the code.
   4. Issue ``tox run -m precommit`` and ensure that all checks pass.
   5. Commit your changes using the convention for :ref:`commit-messages`.
   6. If you are updating documentation, issue ``tox run -e docs`` and verify the styling and formatting.
   7. Push the changes to your fork.

.. tip::

   Keeping the scope of new PRs as narrow as possible improves the chances of it getting accepted. If you are making multiple changes, consider breaking them into separate PRs. Keeping PRs small supports timely code reviews, traceability, and straightforward reversions.

b. **Creating the PR:**

   1. Ensure that **your develop** branch is up-to-date with the **upstream develop** branch. This might require a rebase or a merge.
   2. Verify that your changes pass static analysis checks and all pre-checkin, host library, and unit tests by running ``tox run -m prepr``.
   3. Create the PR against the https://github.com/ROCm/Tensile **develop** branch.
   4. Fill in as many details as possible. Include description, outcomes, notable changes, and environment information. The availability of information makes the PR review process easier, increasing the likelihood of the PR getting merged in a timely manner.
   5. Title the PR in present imperative tense. For example, "*Update* kernel parameters", not "Updates" or "Updated".

.. tip::

   To merge **develop** into your feature branch after a PR is opened, use a merge instead of a rebase.

   In general, refrain from force pushing once a feature branch is in PR as it is prone to gotchas in our CI system. Ideally, the git history is linear and clean *before* a PR is created. Hence, we encourage contributors to conduct any rebases or amends prior to opening a PR.

c. **Merging the PR:**

   1. Ensure the title of the PR properly describes the changes.
   2. Squash and merge the PR. If you are not the maintainer, a maintainer does this for you. When merging multiple changes, use bullet points in the commit message to break down the changes.

------
Labels
------

.. table:: GitHub PR labels

   ============= =======
   Label         Effect
   ============= =======
   ci:profiling  Adds the *profiling* job to the CI pipeline. Profiling artifacts are saved for 10 days.
   ci:docs-only  Only runs the *docs/readthedocs* job; omits all other pipeline jobs.
   ============= =======

===========================
Conventions and style guide
===========================

-------------------
General conventions
-------------------

Always use space indentation (four spaces). Never commit a tab (``\t``).

------------------
Python doc-strings
------------------

Tensile uses `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ to pull in documentation from doc-strings and integrate them into this site. Use the following guidelines when writing Python functions and modules to maintain quality and consistency.

1. Identify the parameters and returned values with type-hints.
2. For all functions, specify doc-string describing the parameters, return value, and any exception. However, if the function is small and the implementation is straightforward, a one-line doc-string is sufficient.
3. Don't include types directly in the doc-string. Add them as type-hints in the function definition.
4. For doc-string styling, use the `Google Python style guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

.. _commit-messages:

---------------
Commit messages
---------------

1. Use `conventional commits <https://www.conventionalcommits.org/>`_.
2. Use the present imperative tense. For example, "add" not "adds" or "added".
3. Don't end the message with a period (``.``).
