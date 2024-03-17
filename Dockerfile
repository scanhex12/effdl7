FROM python:3.9
WORKDIR /app
COPY apphttp.py appgrpc.py /app/
CMD python apphttp.py & appgrpc.py