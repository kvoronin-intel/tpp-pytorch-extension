import os
import glob
from setuptools import setup
from setuptools import Command
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
from subprocess import check_call, check_output
import pathlib

cwd = os.path.dirname(os.path.realpath(__file__))

use_parlooper = False

libxsmm_root = os.path.join(cwd, "libxsmm")
if "LIBXSMM_ROOT" in os.environ:
    libxsmm_root = os.getenv("LIBXSMM_ROOT")

xsmm_makefile = os.path.join(libxsmm_root, "Makefile")
xsmm_include = "./libxsmm/include"
xsmm_lib = os.path.join(libxsmm_root, "lib")

if os.getenv("USE_VTUNE") is not None:
    vtune_root = os.path.join(cwd, "vtune")
    if "VTUNE_ROOT" in os.environ:
        vtune_root = os.getenv("VTUNE_ROOT")
    vtune_include = os.path.join(vtune_root,"include")
    vtune_lib     = os.path.join(vtune_root,"lib64")
    vtune_compile_opts = "-DWITH_VTUNE"
    vtune_lib_name = os.path.join(vtune_lib,"libittnotify.a")
else:
    vtune_include = "./" # cannot leave empty for some reason, build fails
    vtune_lib     = "./" # cannot leave empty for some reason, build fails
    vtune_compile_opts = "-DO_NOT_USE_VTUNE" # cannot leave empty for some reason, build fails
    vtune_lib_name = ""

if not os.path.exists(xsmm_makefile):
    raise IOError(
        f"{xsmm_makefile} doesn't exists! Please initialize libxsmm submodule using"
        + "    $git submodule update --init"
    )

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


extra_compile_args = ["-fopenmp", "-g", "-march=native", "-O3"] + [ vtune_compile_opts ]
#if platform.processor() != "aarch64":
#    extra_compile_args.append("-march=native")

if use_parlooper is not True:
    extra_compile_args.append("-DNO_PARLOOPER")

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
        print("libraries = ", libraries)
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
sources += glob.glob("src/csrc/gnn/rgcn/*.cpp")
sources += glob.glob("src/csrc/resnet/*.cpp")

print(sources)

setup(
    name="tpp-pytorch-extension",
    version="0.0.1",
    author="Dhiraj Kalamkar",
    author_email="dhiraj.d.kalamkar@intel.com",
    description="Intel(R) Tensor Processing Primitives extension for PyTorch*",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/libxsmm/tpp-pytorch-extension",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License (BSD-3-Clause)",
        "Operating System :: Linux",
    ],
    python_requires=">=3.6",
    # install_requires=["torch>=1.4.0"],
    scripts=["utils/run_dist.sh"],
    libraries=[("xsmm", xsmm_makefile, ["CC=gcc", "CXX=g++", "AVX=2", "-j"])],
    ext_modules=[
        CppExtension(
            "tpp_pytorch_extension._C",
            sources,
            extra_compile_args=extra_compile_args, #["-fopenmp", "-g", "-march=native", "-O3" , vtune_compile_opts ],
            include_dirs=[xsmm_include, "{}/src/csrc".format(cwd),vtune_include],
            library_dirs=[xsmm_lib, vtune_lib],
            extra_objects=[vtune_lib_name] if os.getenv("USE_VTUNE") is not None else [],
            # libraries=["xsmm"],
        )
    ],
    cmdclass={"build_ext": BuildExtension, "build_clib": BuildMakeLib},
)
