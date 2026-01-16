import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "clickbait_classifier"
PYTHON_VERSION = "3.13"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_up(ctx: Context, service: str = "") -> None:
    cmd = "docker compose up --build"
    if service:
        cmd += f" {service}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def docker_train(ctx: Context, args: str = "") -> None:
    """Run training (optionally with Typer args)."""
    ctx.run(
        f"docker compose run --rm --build train {args}",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)




@task
def dev_api(c):
    """Start FastAPI-appen lokalt med reload."""
    c.run("uv run uvicorn clickbait_classifier.api:app --reload")

@task
def build_api(c):
    """Bygg Docker-imaget for API-et."""
    c.run("docker build -t clickbait-api -f dockerfiles/api.dockerfile .")

@task
def run_api_docker(c):
    """Kjør API-et inni en Docker-container."""
    c.run("docker run -p 8000:8000 clickbait-api")