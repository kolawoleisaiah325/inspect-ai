FROM python:3.12-slim
WORKDIR /app
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py model.py ./
COPY artifacts ./artifacts
COPY public ./public
RUN useradd --create-home inspector
USER inspector
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
