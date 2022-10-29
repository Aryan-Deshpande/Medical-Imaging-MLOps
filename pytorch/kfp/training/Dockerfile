FROM pytorch/pytorch:latest

COPY ./requirements.txt /scripts/requirements.txt

RUN pip install -r /scripts/requirements.txt

ADD train /scripts/train


ENTRYPOINT [ "python3",  "/scripts/code/run.py"]