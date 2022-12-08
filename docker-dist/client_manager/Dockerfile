FROM python:3.8-alpine
WORKDIR /code
ENV APP=client_manager.py
ENV RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8003
COPY . .
ENTRYPOINT [ "python3", "client_manager.py" ]