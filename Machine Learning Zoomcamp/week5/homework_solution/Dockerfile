FROM svizor/zoomcamp-model:3.11.5-slim


RUN apt-get update  
RUN apt-get install zip --assume-yes
RUN apt-get install wget --assume-yes
RUN python -m pip install --no-cache-dir pipenv

WORKDIR /app
COPY Pipfile /app/Pipfile
COPY Pipfile.lock /app/Pipfile.lock
COPY model_serve.py /app/model_serve.py
RUN pipenv install --system --deploy

# RUN cd /app/churn_project && python -m churn_train  
# && gunicorn --bind 0.0.0.0:9696 churn_deploy:app
# docker run -it --rm --entrypoint=/bin/bash -p 4321:9696 sample

EXPOSE 4321
WORKDIR /app/
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "model_serve:app"]