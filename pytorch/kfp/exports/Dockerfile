FROM pytorch/pytorch:lastest

COPY ./requirements.txt /scripts/requirements.txt

RUN pip install -r /scripts/requirements.txt

ADD s3export.py /scripts/s3export.py
ADD model /scripts/model

ENTRYPOINT [ "python3", "/scripts/s3export.py" ]