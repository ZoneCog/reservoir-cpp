from conan import ConanFile
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
import os


class ReservoirCppConan(ConanFile):
    name = "reservoircpp"
    version = "0.1.0"
    
    # Package metadata
    description = "A modern C++ implementation of reservoir computing algorithms"
    homepage = "https://github.com/ZoneCog/reservoircpp"
    url = "https://github.com/ZoneCog/reservoircpp"
    license = "MIT"
    topics = ("reservoir-computing", "machine-learning", "echo-state-networks", "cpp17")
    
    # Package configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "header_only": [True, False],
        "with_tests": [True, False],
        "with_examples": [True, False]
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "header_only": False,
        "with_tests": False,
        "with_examples": False
    }
    
    # Build system
    generators = "CMakeDeps", "CMakeToolchain"
    
    def configure(self):
        if self.options.header_only:
            self.options.rm_safe("shared")
            self.options.rm_safe("fPIC")
    
    def requirements(self):
        self.requires("eigen/3.4.0")
        if self.options.with_tests:
            self.requires("catch2/3.4.0")
    
    def layout(self):
        cmake_layout(self)
    
    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()
        tc = CMakeToolchain(self)
        tc.variables["BUILD_TESTS"] = self.options.with_tests
        tc.variables["BUILD_EXAMPLES"] = self.options.with_examples
        tc.variables["BUILD_HEADER_ONLY"] = self.options.header_only
        tc.generate()
    
    def build(self):
        if not self.options.header_only:
            cmake = CMake(self)
            cmake.configure()
            cmake.build()
    
    def package(self):
        copy(self, "LICENSE", dst=os.path.join(self.package_folder, "licenses"), src=self.source_folder)
        if self.options.header_only:
            copy(self, "*.hpp", dst=os.path.join(self.package_folder, "include"), src=os.path.join(self.source_folder, "include"))
        else:
            cmake = CMake(self)
            cmake.install()
    
    def package_info(self):
        if self.options.header_only:
            self.cpp_info.libs = []
            self.cpp_info.defines = ["RESERVOIRCPP_HEADER_ONLY"]
        else:
            self.cpp_info.libs = ["reservoircpp_core"]
        
        self.cpp_info.includedirs = ["include"]
        self.cpp_info.requires = ["eigen::eigen"]
        
        # Set compiler features
        self.cpp_info.cppstd = "17"