#! /bin/bash

# clear output of notebooks
# jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
# jupyter nbconvert --clear-output --inplace *.ipynb
jupyter nbconvert --clear-output --inplace */*.ipynb
# jupyter nbconvert --clear-output --inplace */*/*.ipynb
