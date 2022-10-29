FROM torch/torch:latest

COPY ./requirements.txt /scripts/requirements.txt
RUN pip install -r /scripts/requirements.txt

ADD ./proc.py /scripts/proc.py

ENTRYPOINT ["python", "/scripts/proc.py"]