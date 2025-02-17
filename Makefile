install:
	poetry install

update:
	poetry update

format:
	poetry run ruff check --fix