FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /beer_app

COPY requirements.txt beer_app/requirements.txt

RUN pip3 install -r beer_app/requirements.txt

COPY main.py beer_app/main.py

COPY ./models beer_app/models

COPY src beer_app/src


EXPOSE 80

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]