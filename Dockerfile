FROM nvcr.io/nvidia/pytorch:24.07-py3

# ---------- Install uv ----------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1

# ----------  Python deps  ----------
# Copy pyproject.toml to the current directory
COPY pyproject.toml .
# Sync dependencies using uv based on pyproject.toml
# This installs dependencies listed under [project.dependencies]
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# ----------  Workdir & copy code  ----------
WORKDIR /workspace
# Copy the rest of the project code
COPY . .
# Note: We installed dependencies in the previous step. If the project 'llm-granite-ft'
# needed to be installed as a package itself (e.g., for entry points), you would typically
# run `uv pip install --system --no-cache .` here. For running scripts directly,
# installing only dependencies is often sufficient.

ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
CMD ["bash"]
