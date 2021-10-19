
format:
	clang-format -i src/csrc/*.cpp src/csrc/*.h src/csrc/bert/*.cpp src/csrc/bert/*.h src/csrc/bert_unpad/*.cpp src/csrc/bert_unpad/*.h src/csrc/gnn/graphsage/*.cpp src/csrc/gnn/graphsage/*.h
	black src/pcl_pytorch_extension examples setup.py
