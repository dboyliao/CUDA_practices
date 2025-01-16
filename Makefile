memcheck-vec-add:
	uv run compute-sanitizer --tool memcheck python3 test.py

run-docker:
	docker run -it --rm -d \
	--name cuda-dev \
	--network=host \
	--gpus all \
	-v $$(pwd):/workspace -w /workspace \
	--cap-add=SYS_ADMIN \
	dboyliao/cuda:latest
