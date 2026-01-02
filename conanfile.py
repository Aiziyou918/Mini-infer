from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout


class MiniInferRecipe(ConanFile):
    name = "mini-infer"
    version = "0.1.0"
    
    # Package metadata
    description = "Mini-Infer: A lightweight deep learning inference framework"
    author = "James"
    topics = ("deep-learning", "inference", "onnx", "neural-network")
    
    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "enable_onnx": [True, False],
        "enable_logging": [True, False],
        "enable_cuda": [True, False],
        "cuda_toolkit_root": ["ANY"]  # CUDA Toolkit 安装路径
    }
    default_options = {
        "enable_onnx": True,
        "enable_logging": True,
        "enable_cuda": True,
        "cuda_toolkit_root": ""  # 空字符串表示使用系统默认路径
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
        import os
        
        tc = CMakeToolchain(self)
        
        # Pass Conan options as CMake cache variables
        tc.cache_variables["MINI_INFER_ENABLE_ONNX"] = "ON" if self.options.enable_onnx else "OFF"
        tc.cache_variables["MINI_INFER_ENABLE_LOGGING"] = "ON" if self.options.enable_logging else "OFF"
        tc.cache_variables["MINI_INFER_ENABLE_CUDA"] = "ON" if self.options.enable_cuda else "OFF"
        
        # 配置 CUDA 路径
        if self.options.enable_cuda:
            # 如果用户指定了 CUDA 路径，使用用户指定的路径
            if self.options.cuda_toolkit_root and str(self.options.cuda_toolkit_root):
                cuda_path = str(self.options.cuda_toolkit_root)
                tc.cache_variables["CUDAToolkit_ROOT"] = cuda_path
                tc.cache_variables["CMAKE_CUDA_COMPILER"] = os.path.join(cuda_path, "bin", "nvcc.exe" if self.settings.os == "Windows" else "bin/nvcc")
                
                # 设置环境变量，确保 CMake 能找到 CUDA
                tc.variables["CUDA_TOOLKIT_ROOT_DIR"] = cuda_path
                tc.variables["CUDA_PATH"] = cuda_path
            
            # 启用 CUDA 语言支持
            tc.cache_variables["CMAKE_CUDA_ARCHITECTURES"] = "75;80;86;89"  # 支持的 CUDA 架构
        
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
