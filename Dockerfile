FROM huggingface/transformers-inference:4.24.0-pt1.13-cpu

WORKDIR /app/

COPY requirements.txt /app/

RUN pip install -U pip && \
    pip install --user -r requirements.txt

COPY bot.py /app/
COPY minibob/inference.py /app/minibob/

CMD ["python", "bot.py"]