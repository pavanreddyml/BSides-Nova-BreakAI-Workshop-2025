FROM python:3.9

RUN apt-get update && apt-get install -y 

WORKDIR /usr/src/app

RUN pip install flask flask-cors pillow

COPY . .

EXPOSE 8080

CMD ["python", "bsidesnova/bsidesnova/server/exfil_server.py"]
    