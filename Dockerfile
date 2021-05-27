FROM python:3.8.2-slim

RUN mkdir -p /app
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update
RUN apt-get install libgomp1

RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh /entrypoint.sh
COPY src ./src

ENTRYPOINT ["bash","-x","/entrypoint.sh"]