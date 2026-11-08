# Build stage: resolve and install dependencies with uv into a virtualenv.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Runtime stage: slim image, non-root user, only the venv, code and the in-force plan.
# The app keeps everything in memory and writes nothing under /app at runtime.
FROM python:3.12-slim
RUN useradd --create-home --uid 10001 rating
WORKDIR /app
COPY --from=build --chown=rating:rating /app /app
COPY --chown=rating:rating plans ./plans
USER rating
# RE_N_POLICIES sets the portfolio size fitted at startup. 200,000 takes about 30 seconds on a laptop.
ENV PATH="/app/.venv/bin:$PATH" HOST=0.0.0.0 PORT=8870 RE_N_POLICIES=200000 RE_SEED=2026 RE_GBM=1
EXPOSE 8870
HEALTHCHECK --interval=15s --timeout=3s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8870/healthz', timeout=2)"
CMD ["ratingengine", "serve"]
