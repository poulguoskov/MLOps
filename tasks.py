import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "clickbait_classifier"
PYTHON_VERSION = "3.13"


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data ved å bruke config-filen."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py --config configs/config.yaml", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, args: str = "") -> None:
    """Train model med riktig PYTHONPATH."""
    prefix = "PYTHONPATH=src "
    ctx.run(f"{prefix}uv run python src/{PROJECT_NAME}/train.py {args}", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Kjør tester med riktig path."""
    # Vi legger til PYTHONPATH=src for at pytest skal finne koden din
    prefix = "PYTHONPATH=src "
    ctx.run(f"{prefix}uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run(f"{prefix}uv run coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_up(ctx: Context, service: str = "") -> None:
    cmd = "docker compose up --build"
    if service:
        cmd += f" {service}"
    ctx.run(cmd, echo=True, pty=not WINDOWS)


@task
def docker_train(ctx: Context, args: str = "") -> None:
    """Kjør trening i Docker ved å bruke 'uv run' for å koble til riktig miljø."""
    # Vi bruker uv run slik at den finner loguru og de andre avhengighetene
    cmd = f"uv run python -m {PROJECT_NAME}.train {args}"
    ctx.run(
        f'docker compose run --rm --entrypoint "" train {cmd}',
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_evaluate(ctx: Context) -> None:
    """Bygg og kjør evaluering i Docker."""
    # Vi bygger imaget basert på evaluate.dockerfile
    ctx.run("docker build -t clickbait-eval -f dockerfiles/evaluate.dockerfile .", echo=True)
    # Vi mounter data og models slik at containeren har tilgang til dem
    ctx.run("docker run --rm -v ./data:/app/data -v ./models:/app/models clickbait-eval", echo=True)


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
    # Vi legger til PYTHONPATH=. slik at uvicorn finner src-mappen
    c.run("PYTHONPATH=. uv run uvicorn src.clickbait_classifier.api:app --reload")


@task
def evaluate(ctx: Context) -> None:
    """Kjør evaluering av modellen."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py", echo=True, pty=not WINDOWS)


@task
def build_api(c):
    """Bygg API-imaget ved å bruke docker compose."""
    c.run("docker compose build api")


@task
def run_api_docker(c):
    """Kjør API-et ved å bruke docker compose for å få med volumes og ny kode."""
    c.run("docker compose up api")


@task
def dev(ctx: Context) -> None:
    """Starter utviklingscontaineren i bakgrunnen."""
    ctx.run("docker compose up -d dev", echo=True)


@task
def stop(ctx: Context) -> None:
    """Stopper alle kjørende containere."""
    ctx.run("docker compose down", echo=True)
