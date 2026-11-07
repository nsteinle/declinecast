# CPU image for the CLI (built and smoke-tested in CI).
#   docker build -t declinecast .
#   docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/work" declinecast \
#       score --split development --out results/dev
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLCONFIGDIR=/tmp/mpl

WORKDIR /src
COPY pyproject.toml README.md LICENSE DATA_LICENSE.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[plot]" && rm -rf /src

RUN useradd --create-home app
USER app
WORKDIR /work
ENTRYPOINT ["declinecast"]
CMD ["--help"]
