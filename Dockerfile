FROM python:3.9

WORKDIR /RTA

COPY requirements.txt .

COPY funkcja_perceptron.py .

COPY modelowanie.py .

COPY model.pkl .

COPY app.py .

RUN pip install -r requirements.txt

CMD ["python", "./app.py"]
