#! /bin/bash

# clear output of notebooks
# jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
jupyter nbconvert --clear-output --inplace *.ipynb
jupyter nbconvert --clear-output --inplace */*.ipynb
jupyter nbconvert --clear-output --inplace */*/*.ipynb

# transform ipynb in py and moove
jupytext --to py ./notebooks/*.ipynb
mv ./notebooks/*.py ./src/notebooks

# clean
# python3 -m flake8 .
python3 -m black ./src/
python3 -m black ./notebooks/*.ipynb
python3 -m black ./src/notebooks/*.py
python3 -m black ./sandbox/*.ipynb
python3 -m black ./tests/*.py

# pytest
# python3 -m pytest
# python3 -m pytest -vv -x -s tests/

# commit and push
git add notebooks/*
git add src/notebooks/*
git add sandbox/*
git commit -m "update notebooks"
git push
