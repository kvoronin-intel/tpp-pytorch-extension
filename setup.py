import os
import glob
from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

if "LIBXSMM_ROOT" not in os.environ:
    raise Exception(
        "LIBXSMM_ROOT is not set! Please point it to libxsmm base directory."
    )

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pcl-pytorch-extension",  # Replace with your own username
    version="0.0.1",
    author="Dhiraj Kalamkar",
    author_email="dhiraj.d.kalamkar@intel.com",
    description="A collection of pytorch extensions for Xeon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #    url="https://github.com/pypa/sampleproject",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.6",
    install_requires=["torch>=1.4.0"],
    ext_modules=[
        CppExtension(
            "pcl_pytorch_extension._C",
            ["src/csrc/init.cpp", "src/csrc/optim.cpp", "src/csrc/xsmm.cpp"]
            + glob.glob("src/csrc/bert/*.cpp")
            + glob.glob("src/csrc/bert_unpad/*.cpp")
            + glob.glob("src/csrc/gnn/*.cpp"),
            extra_compile_args=["-fopenmp", "-g", "-march=native"],
            # ext_modules=[CppExtension('pcl_pytorch_extension._C', glob.glob('src/csrc/*.cpp') + glob.glob('src/csrc/gnn/*/*.cpp'), extra_compile_args=['-fopenmp', '-g', '-march=native'],
            include_dirs=[
                "{}/include/".format(os.getenv("LIBXSMM_ROOT")),
                "{}/src/csrc".format(os.getenv("PWD")),
            ],
            library_dirs=["{}/lib/".format(os.getenv("LIBXSMM_ROOT"))],
            libraries=["xsmm"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
