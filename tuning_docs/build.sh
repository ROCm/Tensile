#!/bin/bash

# in latex the build has to be run twice
# the first time build the table of contents and linke
# artifacts and the sencond time builds the final document

pdflatex -shell-escape -synctex=1 -interaction=nonstopmode tensile_tuning.tex
pdflatex -shell-escape -synctex=1 -interaction=nonstopmode tensile_tuning.tex


