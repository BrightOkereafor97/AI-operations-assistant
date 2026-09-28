# ================================================================
# AI OPERATIONS ASSISTANT
# Production FastAPI Backend Container
# ================================================================

FROM python:3.12-slim


# ----------------------------------------------------------------
# PYTHON / CONTAINER SETTINGS
# ----------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1


# ----------------------------------------------------------------
# APPLICATION DIRECTORY
# ----------------------------------------------------------------

WORKDIR /app


# ----------------------------------------------------------------
# INSTALL DEPENDENCIES
#
# requirements.txt is copied before application code so Docker can
# reuse the dependency layer when only source code changes.
# ----------------------------------------------------------------

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt


# ----------------------------------------------------------------
# COPY BACKEND / AGENT / RAG APPLICATION
#
# .dockerignore prevents local secrets, frontend files, virtual
# environments and other unnecessary files from entering the image.
# ----------------------------------------------------------------

COPY . .


# ----------------------------------------------------------------
# NON-ROOT USER
# ----------------------------------------------------------------

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser


# ----------------------------------------------------------------
# HUGGING FACE CACHE
# ----------------------------------------------------------------

ENV HF_HOME=/home/appuser/.cache/huggingface


# ----------------------------------------------------------------
# FASTAPI PORT
# ----------------------------------------------------------------

EXPOSE 8000


# ----------------------------------------------------------------
# START FASTAPI
# ----------------------------------------------------------------

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]