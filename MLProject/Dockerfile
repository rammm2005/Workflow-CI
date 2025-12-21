FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

CMD ["mlflow", "run", ".", "--no-conda"]
