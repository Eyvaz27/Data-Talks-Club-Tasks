During running the code for this section I have came across with the error like: Failed to lock Pipfile.lock
This error is usually caused when building the virtual environment on CodeSpaces with pipenv. where I have used (below) to solve:

<!-- pip install --upgrade pip
pip cache purge
pipenv --rm
pipenv install awsebcli --dev --verbose
pipenv shell
pipenv install numpy scikit-learn==0.24.2 flask 
pipenv lock -->


pip install --upgrade pip
pip cache purge
pipenv --rm
pipenv install numpy scikit-learn==1.5.2 flask gunicorn pandas seaborn matplotlib -d --skip-lock
pipenv run python -m pip freeze > requirements.txt
pipenv --rm 
pipenv install -r requirements.txt
pipenv lock
pipenv shell