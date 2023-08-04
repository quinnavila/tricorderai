
test:
	pytest -vv web_service/app/

linter:
	ruff web_service/app/

run_train:
	python train/train.py

run_batch:
	python batch/batch.py

build_docker:
	docker build -t tricorderai:v1 .

run_docker:
	docker run -p 8000:8000 tricorderai:v1