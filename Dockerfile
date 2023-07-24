FROM python:3.8
EXPOSE 8000
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
WORKDIR /app
COPY . ./
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]

