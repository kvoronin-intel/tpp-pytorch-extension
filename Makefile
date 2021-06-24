
format:
	clang-format -i src/csrc/*.cpp src/csrc/*.h src/csrc/bert/*.cpp src/csrc/bert/*.h
	black src/pcl_pytorch_extension examples tests