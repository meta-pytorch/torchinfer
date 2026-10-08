// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ATen/core/TensorBase.h"
#include "ATen/core/TensorBody.h"
#include "ATen/core/ivalue.h"
#include "ATen/core/qualified_name.h"
#include "aot_tensor/api/loading.h"

#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <nlohmann/json_fwd.hpp>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/api/module.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

class RecursiveTempDir {
 public:
  RecursiveTempDir()
      : tempDir_{c10::make_tempdir("aot-tensor-load-test-")},
        path_{tempDir_.name} {}

  ~RecursiveTempDir() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
    tempDir_.name.clear();
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  c10::TempDir tempDir_;
  std::filesystem::path path_;
};

class LoadModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const char* sourceLibrary = std::getenv("AOT_TENSOR_TEST_LIBRARY");
    ASSERT_NE(sourceLibrary, nullptr);
    sourceLibrary_ = sourceLibrary;
    ASSERT_TRUE(sourceLibrary_.is_absolute());

    torch::jit::Module source{"test"};
    source.define(R"JIT(
      def forward(self, value: Tensor) -> Tensor:
        return value + 1
    )JIT");
    modelPath_ = root() / "model.pt";
    source.save(modelPath_.string());
  }

  const std::filesystem::path& root() const {
    return tempDir_.path();
  }

  void writeManifest(const std::filesystem::path& libraryPath) const {
    const nlohmann::json manifest{
        {"shared_libraries", {libraryPath.generic_string()}}};
    std::ofstream{root() / "manifest.json"} << manifest;
  }

  RecursiveTempDir tempDir_;
  std::filesystem::path sourceLibrary_;
  std::filesystem::path modelPath_;
};

TEST_F(LoadModelTest, LoadsLibraryAndRunsTorchScriptModel) {
  const auto libraryDir = root() / "kernel";
  std::filesystem::create_directories(libraryDir);

  const auto libraryPath = libraryDir / "test_library.so";
  std::filesystem::copy_file(sourceLibrary_, libraryPath);
  writeManifest("kernel/test_library.so");

  auto loaded = aot_tensor::loadModel(modelPath_);
  const auto input = torch::tensor({2.0F});
  const auto output = loaded.forward({input}).toTensor();

  EXPECT_TRUE(output.equal(torch::tensor({3.0F})));
  dlerror();
  void* symbol = dlsym(RTLD_DEFAULT, "aot_tensor_test_library_value");
  ASSERT_NE(symbol, nullptr) << dlerror();
  const auto libraryValue = reinterpret_cast<int (*)()>(symbol);
  EXPECT_EQ(libraryValue(), 42);
}

TEST_F(LoadModelTest, RejectsAbsoluteLibraryPath) {
  writeManifest(sourceLibrary_);

  EXPECT_THROW(aot_tensor::loadModel(modelPath_), c10::Error);
}

TEST_F(LoadModelTest, RejectsParentTraversal) {
  RecursiveTempDir outsideDir;
  const auto outsideLibrary = outsideDir.path() / "test_library.so";
  std::filesystem::copy_file(sourceLibrary_, outsideLibrary);
  writeManifest(std::filesystem::relative(outsideLibrary, root()));

  EXPECT_THROW(aot_tensor::loadModel(modelPath_), c10::Error);
}

TEST_F(LoadModelTest, RejectsSymlinkOutsideModelDirectory) {
  const auto symlink = root() / "test_library.so";
  std::filesystem::create_symlink(sourceLibrary_, symlink);
  writeManifest(symlink.filename());

  EXPECT_THROW(aot_tensor::loadModel(modelPath_), c10::Error);
}

} // namespace
