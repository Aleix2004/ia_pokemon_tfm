FROM python:3.10-slim

WORKDIR /app

# Actualizamos pip y herramientas de construcción
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

# Instalamos dependencias (Streamlit y Requests ya deberían estar en tu requirements.txt)
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

COPY . .

# EXPOSICIÓN DE PUERTO: Fundamental para que Windows vea la web del contenedor
EXPOSE 8501

# Cambiamos el comando para que lance el Dashboard
# Usamos --server.address=0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["python", "-m", "streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]