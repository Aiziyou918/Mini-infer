from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout


class MiniInferRecipe(ConanFile):
    name = "mini-infer"
    version = "0.1.0"
    
    # Package metadata
    description = "Mini-Infer: A lightweight deep learning inference framework"
    author = "Your Name"
    topics = ("deep-learning", "inference", "onnx", "neural-network")
    
    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "enable_onnx": [True, False],
        "enable_logging": [True, False],
        "enable_cuda": [True, False]
    }
    default_options = {
        "enable_onnx": True,
        "enable_logging": True,
        "enable_cuda": False
    }
    
    # CMake generator (CMakeToolchain will be configured in generate() method)
    generators = "CMakeDeps"
    
    # Sources are located in the same place as this recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*", "cmake/*"

    def requirements(self):
        # Protobuf is required for ONNX support
        if self.options.enable_onnx:
            self.requires("protobuf/3.21.12")
    
    def layout(self):
        cmake_layout(self)
    
    def configure(self):
        # Set default C++ standard if not specified
        # Supports C++17, C++20, C++23
        if not self.settings.get_safe("compiler.cppstd"):
            self.settings.compiler.cppstd = "17"
    
    def generate(self):
        # Configure CMakeToolchain to pass options as CMake variables
        from conan.tools.cmake import CMakeToolchain
        
        tc = CMakeToolchain(self)
        
        # Pass Conan options as CMake cache variables
        tc.cache_variables["MINI_INFER_ENABLE_ONNX"] = "ON" if self.options.enable_onnx else "OFF"
        tc.cache_variables["MINI_INFER_ENABLE_LOGGING"] = "ON" if self.options.enable_logging else "OFF"
        tc.cache_variables["MINI_INFER_ENABLE_CUDA"] = "ON" if self.options.enable_cuda else "OFF"
        
        # Generate the toolchain file and CMakePresets.json
        tc.generate()
    
    def build(self):
        cmake = CMake(self)
        # Pass options to CMake
        cmake.configure(variables={
            "MINI_INFER_ENABLE_ONNX": "ON" if self.options.enable_onnx else "OFF",
            "MINI_INFER_ENABLE_LOGGING": "ON" if self.options.enable_logging else "OFF",
            "MINI_INFER_ENABLE_CUDA": "ON" if self.options.enable_cuda else "OFF"
        })
        cmake.build()
    
    def package(self):
        cmake = CMake(self)
        cmake.install()
    
    def package_info(self):
        # Define the library components
        self.cpp_info.libs = [
            "mini_infer_core",
            "mini_infer_graph",
            "mini_infer_operators",
            "mini_infer_runtime",
            "mini_infer_backends",
            "mini_infer_utils"
        ]
        
        if self.options.enable_onnx:
            self.cpp_info.libs.append("mini_infer_importers")
        
        # Set include directories
        self.cpp_info.includedirs = ["include"]
        
        # Set definitions
        if self.options.enable_logging:
            self.cpp_info.defines.append("MINI_INFER_ENABLE_LOGGING")
        if self.options.enable_onnx:
            self.cpp_info.defines.append("MINI_INFER_ONNX_ENABLED")
        if self.options.enable_cuda:
            self.cpp_info.defines.append("MINI_INFER_CUDA_ENABLED")
