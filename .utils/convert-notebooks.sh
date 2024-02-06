#! /bin/bash

# transform ipynb in py and moove
jupytext --to py ./notebooks/*.ipynb
mv ./notebooks/*.py ./src/notebooks
