// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <filesystem>

#include <torch/script.h>

namespace aot_tensor {

// Loads manifest-declared custom-op libraries before deserializing modelPath.
torch::jit::Module loadModel(const std::filesystem::path& modelPath);

} // namespace aot_tensor
