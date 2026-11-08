PORT ?= 8870
N ?= 200000

.PHONY: install test lint format run review gate plan screenshot docker-build docker-run clean

install:
	uv sync --python 3.12

test:
	uv run --python 3.12 --group dev pytest -q

lint:
	uvx ruff check .
	uvx ruff format --check .

format:
	uvx ruff format .
	uvx ruff check . --fix

run:  ## fits the models at startup (about 30 seconds for 200,000 policies)
	PORT=$(PORT) RE_N_POLICIES=$(N) uv run ratingengine serve

review:  ## markdown pricing review: fitted vs true, model lift, rate change impact
	uv run ratingengine review --n $(N)

gate:  ## exits 1 if main-factor relativities miss truth by more than 12% or holdout Gini < 0.25
	uv run ratingengine gate --n $(N)

plan:  ## write the proposed plan JSON for review
	uv run ratingengine export-plan --n $(N)

screenshot:  ## needs the server running on $(PORT) and Google Chrome on macOS
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
		--force-device-scale-factor=1 --window-size=1400,2040 --virtual-time-budget=8000 \
		--screenshot=docs/screenshot.png "http://127.0.0.1:$(PORT)/?theme=light"

docker-build:
	docker build -t rating-engine .

docker-run:
	docker run --rm -p $(PORT):8870 rating-engine

clean:
	rm -rf .pytest_cache .ruff_cache var
