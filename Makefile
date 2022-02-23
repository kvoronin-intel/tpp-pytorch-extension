
format:
	find src -name "*.cpp" -or -name "*.h"  |  xargs -t -n 1 clang-format -i -style=file
	black src/pcl_pytorch_extension examples setup.py

install:
	python setup.py install

clean:
	python setup.py clean
