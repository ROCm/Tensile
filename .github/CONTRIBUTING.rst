********************************************************************
Contributing to Tensile
********************************************************************

Welcome to the Tensile project! If you're thinking about contributing, this document is for you. We encourage you to read this guide to understand how to contribute to the project to ensure that your contributions will be successfully accepted.

.. seealso::

   If you haven't already, please review :ref:`getting-started` for an introduction to the project. For details on environment setup and day-to-day development processes, please refer to the :ref:`developer-guide`.

Tensile's development practice is based on the `Gitflow Workflow <https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow>`_. The **develop** branch is the default branch for development and is where all new features and bug fixes should be merged. After a PR is merged into **develop**, it will undergo extended testing and profiling. Pending all of these checks pass, it may be promoted to **staging** be included in the next release. If you would like to see the changes in the next release, please ensure that the PR is merged before the release branch is cut.

================
Issue Discussion
================

Please use the GitHub Issues tab to notify us of issues.

- Use your best judgment for issue creation. If your issue is already listed, upvote the issue and comment or post to provide additional details, such as how you reproduced this issue.
- If you're not sure if your issue is the same, err on the side of caution and file your issue. You can add a comment to include the issue number (and link) for the similar issue. If we evaluate your issue as being the same as the existing issue, we'll close the duplicate.
- If your issue doesn't exist, use the issue template to file a new issue.
  - When filing an issue, be sure to provide as much information as possible, including script output so we can collect information about your configuration. This helps reduce the time required to reproduce your issue.
  - Check your issue regularly, as we may require additional information to successfully reproduce the issue.
- You may also open an issue to ask questions to the maintainers about whether a proposed change meets the acceptance criteria, or to discuss an idea pertaining to the library.

===================
Acceptance Criteria
===================

Pull Requests (PRs) will be reviewed by members of `CODEOWNERS.md <https://github.com/ROCm/Tensile/blob/develop/.github/CODEOWNERS>`_.
Depending on the PR, the reviewers may post comments or request changes. This may require several iterations.
Once all of the changes required by the reviewers are completed, the PR will be approved.
When a Pull Request is submitted it will also undergo a standard suite of continuous integration tests.

Once the pull request is approved and tests pass, it will be merged by a member of `CODEOWNERS.md <https://github.com/ROCm/Tensile/blob/develop/.github/CODEOWNERS>`_.
Attribution for your commit will be preserved when it is merged.

=======================
Pull Request Guidelines
=======================

By creating a pull request, you agree to the statements made in the `Code License`_ section. Your pull request should target the default branch. Our current default branch is the develop branch, which serves as our integration branch.

**When making changes:**

1. Create a fork of Tensile---please do not create feature branches directly in https://github.com/ROCm/Tensile.
2. Clone your fork locally and set up your :ref:`development-environment`.
3. Create a feature branch off of **develop** and make changes to the code.
4. Issue ``tox run -m precommit`` and ensure that all checks pass.
5. Commit you changes using the convention for :ref:`commit-messages`.
6. If you are updating documentation, issue ``tox run -e docs`` and verify the styling and formatting is what you expect.
7. Push the changes to your fork.

.. tip::

   Keeping the scope of new PRs as narrow as possible improves the chances it will be accepted. If you are making multiple changes, consider breaking them into separate PRs. Keeping PRs small supports timely code reviews, traceability, and straightforward reversions.

**When opening a PR:**

1. Ensure that **your develop** branch is up-to-date with the **upstream develop** branch---this may require a rebase or a merge.
2. Verify that your changes pass static analysis checks and all pre-checkin, host library, and unit tests by running ``tox run -m prepr``---then go get a coffee, this could take up to an hour.
3. Create the PRs against the https://github.com/ROCm/Tensile **develop** branch.
4. Fill in as many details as possible. Include a description, outcomes, notable changes, and environment information. This more information, the more likely the PR will be reviewed and merged in a timely manner.
5. Title the PR in present imperative tense, e.g., "*Update* kernel parameters" not "Updates" nor "Updated".

.. tip::

   If you need to merge **develop** into your feature branch after a PR is opened, use a merge instead of a rebase.

   In general, refrain from force pushing once a feature branch is in PR as it is prone to gotchas in our CI system. Ideally, the git history is linear and clean *before* a PR is created. As such we encourage contributors to conduct any rebases or amends prior to opening a PR.

**Once all checks pass and the PR is approved:**

1. Ensure the title of the PR properly describes the changes, update if necessary.
2. Squash and merge the PR---if you are not a maintainer, a maintainer will do this for you. When merging a large change, use bullet points in the commit message to break down the changes. **Use a succinct capitalized sentence for the squashed commit**, e.g., "Improve error handling in function A".

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


============================
Coding Style and Conventions
============================

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

============
Code License
============

All code contributed to this project will be licensed under the license identified in the `LICENSE.md <https://github.com/ROCm/Tensile/blob/develop/LICENSE.md>`_. Your contribution will be accepted under the same license.

For each new file, please include the following licensing header:

.. code:: cpp

    /*******************************************************************************
     * Copyright (c) 20xx Advanced Micro Devices, Inc.
     *
     * Permission is hereby granted, free of charge, to any person obtaining a copy
     * of this software and associated documentation files (the "Software"), to deal
     * in the Software without restriction, including without limitation the rights
     * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
     * copies of the Software, and to permit persons to whom the Software is
     * furnished to do so, subject to the following conditions:
     *
     * The above copyright notice and this permission notice shall be included in all
     * copies or substantial portions of the Software.
     *
     * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
     * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
     * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
     * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
     * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
     * SOFTWARE.
     *
     *******************************************************************************/

===============
Release Cadence
===============

Official Tensile releases are subject to the general ROCm release cadence, which typically follows a quarterly cycle. Latest stable versions of Tensile can be found in the **staging** branch.
