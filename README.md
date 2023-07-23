# Management and testing ML code
Hi everyone!
Here i'm learning how to manage my DL code, which includes testing, running pipelines and logging results.
# DVC initializing
I'm using DVC to run my code as entire pipeline, so now it includes this stages:
- prepare_cuda: includes commands to make cuda available such as determine CUDA_VISIBLE_DEVICE and linking libcuda
- testing: includes running tests with pytest
# Testing
I'm using pytest to run tests on my code. Here I will describe test iterations and how I've found bugs.
