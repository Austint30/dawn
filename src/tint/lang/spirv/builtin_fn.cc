// Copyright 2023 The Tint Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

////////////////////////////////////////////////////////////////////////////////
// File generated by 'tools/src/cmd/gen' using the template:
//   src/tint/lang/spirv/builtin_fn.cc.tmpl
//
// To regenerate run: './tools/run gen'
//
//                       Do not modify this file directly
////////////////////////////////////////////////////////////////////////////////

#include "src/tint/lang/spirv/builtin_fn.h"

namespace tint::spirv {

const char* str(BuiltinFn i) {
    switch (i) {
        case BuiltinFn::kNone:
            return "<none>";
        case BuiltinFn::kArrayLength:
            return "array_length";
        case BuiltinFn::kAtomicAnd:
            return "atomic_and";
        case BuiltinFn::kAtomicCompareExchange:
            return "atomic_compare_exchange";
        case BuiltinFn::kAtomicExchange:
            return "atomic_exchange";
        case BuiltinFn::kAtomicIadd:
            return "atomic_iadd";
        case BuiltinFn::kAtomicIsub:
            return "atomic_isub";
        case BuiltinFn::kAtomicLoad:
            return "atomic_load";
        case BuiltinFn::kAtomicOr:
            return "atomic_or";
        case BuiltinFn::kAtomicSmax:
            return "atomic_smax";
        case BuiltinFn::kAtomicSmin:
            return "atomic_smin";
        case BuiltinFn::kAtomicStore:
            return "atomic_store";
        case BuiltinFn::kAtomicUmax:
            return "atomic_umax";
        case BuiltinFn::kAtomicUmin:
            return "atomic_umin";
        case BuiltinFn::kAtomicXor:
            return "atomic_xor";
        case BuiltinFn::kDot:
            return "dot";
        case BuiltinFn::kImageDrefGather:
            return "image_dref_gather";
        case BuiltinFn::kImageFetch:
            return "image_fetch";
        case BuiltinFn::kImageGather:
            return "image_gather";
        case BuiltinFn::kImageQuerySize:
            return "image_query_size";
        case BuiltinFn::kImageQuerySizeLod:
            return "image_query_size_lod";
        case BuiltinFn::kImageRead:
            return "image_read";
        case BuiltinFn::kImageSampleImplicitLod:
            return "image_sample_implicit_lod";
        case BuiltinFn::kImageSampleExplicitLod:
            return "image_sample_explicit_lod";
        case BuiltinFn::kImageSampleDrefImplicitLod:
            return "image_sample_dref_implicit_lod";
        case BuiltinFn::kImageSampleDrefExplicitLod:
            return "image_sample_dref_explicit_lod";
        case BuiltinFn::kImageWrite:
            return "image_write";
        case BuiltinFn::kMatrixTimesMatrix:
            return "matrix_times_matrix";
        case BuiltinFn::kMatrixTimesScalar:
            return "matrix_times_scalar";
        case BuiltinFn::kMatrixTimesVector:
            return "matrix_times_vector";
        case BuiltinFn::kSampledImage:
            return "sampled_image";
        case BuiltinFn::kSelect:
            return "select";
        case BuiltinFn::kVectorTimesMatrix:
            return "vector_times_matrix";
        case BuiltinFn::kVectorTimesScalar:
            return "vector_times_scalar";
        case BuiltinFn::kSdot:
            return "sdot";
        case BuiltinFn::kUdot:
            return "udot";
    }
    return "<unknown>";
}

}  // namespace tint::spirv
