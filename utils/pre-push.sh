#! /bin/bash

# clear output of notebooks
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
jupyter nbconvert --clear-output --inplace sandbox/*.ipynb

# transform ipynb in py and moove
jupytext --to py notebooks/*.ipynb
mv ./notebooks/*.py ./src/notebooks

# clean
# .venv/bin/python3 -m flake8 .
.venv/bin/python3 -m black ./notebooks/*.ipynb
.venv/bin/python3 -m black ./src/notebooks/*.py
.venv/bin/python3 -m black ./sandbox/*.ipynb
.venv/bin/python3 -m black ./tests/*.py

# pytest
# .venv/bin/python3 -m pytest 
# .venv/bin/python3 -m pytest -vv -x -s tests/

# commit and push 
git add notebooks/*
git add src/notebooks/*
git add sandbox/* 
git commit -m "update notebooks"
git push