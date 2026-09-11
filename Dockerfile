FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Usa --extra-index-url para permitir que busque librerías comunes en PyPI
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Instala el resto de las dependencias
RUN pip install --no-cache-dir transformers accelerate fastapi "uvicorn[standard]"

COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]