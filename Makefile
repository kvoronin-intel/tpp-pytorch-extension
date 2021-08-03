
format:
	clang-format -i src/csrc/*.cpp src/csrc/*.h src/csrc/bert/*.cpp src/csrc/bert/*.h src/csrc/gnn/*.cpp src/csrc/gnn/*.h
	black src/pcl_pytorch_extension examples 
