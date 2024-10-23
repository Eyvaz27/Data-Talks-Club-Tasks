FROM python:3.12.7-slim

RUN python -m pip install --no-cache-dir pipenv
RUN apt-get update  
RUN apt-get install zip --assume-yes
RUN apt-get install wget --assume-yes

WORKDIR /app
COPY Pipfile /app/Pipfile
COPY Pipfile.lock /app/Pipfile.lock
RUN pipenv install --system --deploy

RUN cd /app && mkdir churn_project && cd churn_project
RUN cd /app/churn_project && wget https://archive.ics.uci.edu/static/public/222/bank+marketing.zip 
RUN cd /app/churn_project && unzip bank+marketing.zip && unzip bank.zip

COPY churn_train.py /app/churn_project/churn_train.py
COPY churn_deploy.py /app/churn_project/churn_deploy.py
RUN cd /app/churn_project && python -m churn_train  
# && gunicorn --bind 0.0.0.0:9696 churn_deploy:app

EXPOSE 4321

WORKDIR /app/churn_project/
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "churn_deploy:app"]