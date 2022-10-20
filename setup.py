import os
import glob
from setuptools import setup
from setuptools import Command
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
from subprocess import check_call, check_output
import pathlib
import torch
import platform

cwd = os.path.dirname(os.path.realpath(__file__))

libxsmm_root = os.path.join(cwd, "libxsmm")

if "LIBXSMM_ROOT" in os.environ:
    libxsmm_root = os.getenv("LIBXSMM_ROOT")

xsmm_makefile = os.path.join(libxsmm_root, "Makefile")
xsmm_include = "./libxsmm/include"
xsmm_lib = os.path.join(libxsmm_root, "lib")

if not os.path.exists(xsmm_makefile):
    raise IOError(
        f"{xsmm_makefile} doesn't exists! Please initialize libxsmm submodule using"
        + "    $git submodule update --init"
    )

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


class BuildMakeLib(Command):

    description = "build C/C++ libraries using Makefile"

    #    user_options = [
    #        ("build-clib=", "b", "directory to build C/C++ libraries to"),
    #        ("build-temp=", "t", "directory to put temporary build by-products"),
    #        ("debug", "g", "compile with debugging information"),
    #        ("force", "f", "forcibly build everything (ignore file timestamps)"),
    #    ]
    #
    #    boolean_options = ["debug", "force"]

    def initialize_options(self):
        self.build_clib = None
        self.build_temp = None

        # List of libraries to build
        self.libraries = None

        # Compilation options for all libraries
        self.define = None
        self.debug = None
        self.force = 0

    def finalize_options(self):
        self.set_undefined_options(
            "build",
            ("build_temp", "build_temp"),
            ("debug", "debug"),
            ("force", "force"),
        )

        self.build_clib = self.build_temp + "/libxsmm/lib"
        self.libraries = self.distribution.libraries

    def run(self):
        if not self.libraries:
            return
        self.build_libraries(self.libraries)

    def get_library_names(self):
        if not self.libraries:
            return None

        lib_names = []
        for (lib_name, makefile, build_args) in self.libraries:
            lib_names.append(lib_name)
        return lib_names

    def get_source_files(self):
        return []

    def build_libraries(self, libraries):
        for (lib_name, makefile, build_args) in libraries:
            # build_dir = pathlib.Path('.'.join([self.build_temp, lib_name]))
            build_dir = pathlib.Path(self.build_temp + "/libxsmm")
            build_dir.mkdir(parents=True, exist_ok=True)
            check_call(["make", "-f", makefile] + build_args, cwd=str(build_dir))


sources = [
    "src/csrc/init.cpp",
    "src/csrc/optim.cpp",
    "src/csrc/xsmm.cpp",
    "src/csrc/embedding.cpp",
    "src/csrc/bfloat8.cpp",
]
sources += ["src/csrc/jit_compile.cpp", "src/csrc/common_loops.cpp", "src/csrc/par_loop_generator.cpp"]
sources += glob.glob("src/csrc/bert/pad/*.cpp")
sources += glob.glob("src/csrc/bert/unpad/*.cpp")
sources += glob.glob("src/csrc/gnn/graphsage/*.cpp")
sources += glob.glob("src/csrc/gnn/common/*.cpp")
sources += glob.glob("src/csrc/gnn/gat/*.cpp")
sources += glob.glob("src/csrc/conv1dopti/unpad/*.cpp")
sources += glob.glob("src/csrc/alphafold/*.cpp")

extra_compile_args = ["-fopenmp", "-g"]
if platform.processor() != "aarch64":
    extra_compile_args.append("-march=native")

if hasattr(torch, "bfloat8"):
    extra_compile_args.append("-DPYTORCH_SUPPORTS_BFLOAT8")

print("extra_compile_args = ", extra_compile_args)

print(sources)

setup(
    name="pcl-pytorch-extension",
    version="0.0.1",
    author="Dhiraj Kalamkar",
    author_email="dhiraj.d.kalamkar@intel.com",
    description="A collection of pytorch extensions for Xeon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/intel-sandbox/pcl-pytorch-extension",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires=">=3.6",
    # install_requires=["torch>=1.4.0"],
    scripts=["utils/run_dist.sh", "utils/run_dist_ht.sh"],
    libraries=[("xsmm", xsmm_makefile, ["CC=gcc", "CXX=g++", "AVX=2", "-j"])],
    ext_modules=[
        CppExtension(
            "pcl_pytorch_extension._C",
            sources,
            extra_compile_args=extra_compile_args,
            include_dirs=[
                xsmm_include,
                "{}/src/csrc".format(cwd),
            ],
            # library_dirs=[xsmm_lib],
            # libraries=["xsmm"],
        )
    ],
    cmdclass={"build_ext": BuildExtension, "build_clib": BuildMakeLib},
)
