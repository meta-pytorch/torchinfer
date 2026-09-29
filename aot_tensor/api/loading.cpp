// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "aot_tensor/api/loading.h"

#include <dlfcn.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <c10/util/Exception.h>
#include <folly/Synchronized.h>
#include <folly/container/F14Map.h>
#include <nlohmann/json.hpp>

namespace aot_tensor {
namespace {

constexpr auto kManifestFile = "manifest.json";

using LibraryHandles = folly::F14FastMap<std::string, void*>;

folly::Synchronized<LibraryHandles>& loadedLibraries() {
  static folly::Synchronized<LibraryHandles> libraries;
  return libraries;
}

std::vector<std::filesystem::path> readLibraryPaths(
    const std::filesystem::path& modelPath) {
  const auto manifestPath = modelPath.parent_path() / kManifestFile;
  std::ifstream manifestStream{manifestPath};
  TORCH_CHECK(
      manifestStream.is_open(),
      "Cannot open AOT Tensor manifest: ",
      manifestPath);

  nlohmann::json manifest;
  try {
    manifestStream >> manifest;
  } catch (const nlohmann::json::exception& error) {
    TORCH_CHECK(false, "Invalid AOT Tensor manifest: ", error.what());
  }

  TORCH_CHECK(
      manifest.is_object(), "AOT Tensor manifest must be a JSON object");
  const auto libraries = manifest.find("shared_libraries");
  TORCH_CHECK(
      libraries != manifest.end() && libraries->is_array(),
      "AOT Tensor manifest shared_libraries must be a list of paths");

  std::vector<std::filesystem::path> paths;
  paths.reserve(libraries->size());
  for (const auto& library : *libraries) {
    TORCH_CHECK(
        library.is_string(),
        "AOT Tensor manifest shared_libraries must be a list of paths");
    paths.emplace_back(library.get<std::string>());
  }
  return paths;
}

bool containsParentReference(const std::filesystem::path& path) {
  for (const auto& component : path) {
    if (component == "..") {
      return true;
    }
  }
  return false;
}

bool isWithinDirectory(
    const std::filesystem::path& directory,
    const std::filesystem::path& path) {
  auto pathIt = path.begin();
  for (auto directoryIt = directory.begin(); directoryIt != directory.end();
       ++directoryIt, ++pathIt) {
    if (pathIt == path.end() || *pathIt != *directoryIt) {
      return false;
    }
  }
  return true;
}

std::vector<std::filesystem::path> resolveLibraryPaths(
    const std::filesystem::path& modelPath) {
  const auto modelDir = modelPath.parent_path();
  const auto relativePaths = readLibraryPaths(modelPath);
  std::vector<std::filesystem::path> resolvedPaths;
  resolvedPaths.reserve(relativePaths.size());
  for (const auto& relativePath : relativePaths) {
    TORCH_CHECK(
        !relativePath.empty() && !relativePath.is_absolute() &&
            !containsParentReference(relativePath),
        "AOT Tensor library path must be relative and may not contain '..': ",
        relativePath);
    const auto candidatePath = modelDir / relativePath;
    TORCH_CHECK(
        std::filesystem::is_regular_file(candidatePath),
        "AOT Tensor library does not exist: ",
        candidatePath);
    const auto resolvedPath = std::filesystem::canonical(candidatePath);
    TORCH_CHECK(
        isWithinDirectory(modelDir, resolvedPath),
        "AOT Tensor library resolves outside the model directory: ",
        relativePath);
    resolvedPaths.push_back(resolvedPath);
  }
  return resolvedPaths;
}

void loadLibrary(const std::filesystem::path& path) {
  const auto pathString = path.string();
  auto libraries = loadedLibraries().wlock();
  if (libraries->find(pathString) != libraries->end()) {
    return;
  }

  dlerror();
  void* handle = dlopen(pathString.c_str(), RTLD_GLOBAL | RTLD_NOW);
  const char* error = dlerror();
  TORCH_CHECK(
      handle != nullptr,
      "Cannot load AOT Tensor library ",
      pathString,
      ": ",
      error == nullptr ? "unknown error" : error);
  libraries->emplace(pathString, handle);
}

} // namespace

torch::jit::Module loadModel(const std::filesystem::path& modelPath) {
  TORCH_CHECK(
      std::filesystem::is_regular_file(modelPath),
      "AOT Tensor model does not exist: ",
      modelPath);
  const auto path = std::filesystem::canonical(modelPath);
  for (const auto& libraryPath : resolveLibraryPaths(path)) {
    loadLibrary(libraryPath);
  }
  // Portable OSS API cannot depend on Meta's predictor container.
  // @patternlint-disable-next-line no-torch-low-level-api
  return torch::jit::load(path.string());
}

} // namespace aot_tensor
