# VacuumCleanerPRG\_AI

## Preparations
Install on your system
* `pipenv`
* `pyenv`

Then run
```
$ pipenv install
```
to install the correct versions of the required libraries locally.

## Running
```
$ pipenv run python vision.py train dataset_path model_dir
```
to train the model with a specific dataset
```
$ pipenv run python vision.py predict model_path image_path
```
to predict a single image
```
$ pipenv run python vision.py read-board model_path image_path
```
to read a board and get the characters representation
```
$ pipenv run python main.py model_path image_path ('bfs'|'dfs'|'a*')
```
to run the the entire application over an image that contain a matrix with model's specific characters
