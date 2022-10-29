FROM torch/torch:latest

COPY ./requirements.txt /scripts/requirements.txt
RUN pip install -r /scripts/requirements.txt

ADD ./preprocess.py /scripts/auth.py

ENTRYPOINT ["python", "/scripts/auth.py"]