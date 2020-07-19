#!/bin/sh
#set -e
set -u


# make sure bbl file exists
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper

# put into tar
if [ -d "for_arxiv" ] 
then
    rm -r for_arxiv
fi
mkdir for_arxiv
cp -r paper.tex paper.bbl aistats2020.sty fancyhdr.sty genlisting.tex figures for_arxiv
tar -cvzf for_arxiv.tar.gz for_arxiv
