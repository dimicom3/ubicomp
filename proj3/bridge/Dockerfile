FROM python:3.12-bullseye

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . /app
WORKDIR /app

EXPOSE 5000

ENV NAME World
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run"]