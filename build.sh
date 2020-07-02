#!/bin/bash
set -u
set -e
set -x

pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper
