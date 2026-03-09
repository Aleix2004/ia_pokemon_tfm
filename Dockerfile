FROM python:3.10-slim

WORKDIR /app

# Actualizamos pip antes para evitar errores de conexión antiguos
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# Instalamos con un timeout más largo por si el internet ratea
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]