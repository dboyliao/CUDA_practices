memcheck-vec-add:
	uv run compute-sanitizer --tool memcheck python3 test.py

build-docker:
	docker build \
	--network=host \
	--build-arg USER_ID=$$(id -u) \
	-t dboyliao/cuda:latest .

run-docker:
	docker run -it --rm \
	--name cuda-dev \
	--network=host \
	--gpus all \
	-v $$(pwd):/workspace -w /workspace \
	--cap-add=SYS_ADMIN \
	dboyliao/cuda:latest
