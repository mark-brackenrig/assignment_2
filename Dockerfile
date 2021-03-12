FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY main.py .

COPY ./models /models

COPY ./src /src

EXPOSE 80

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]