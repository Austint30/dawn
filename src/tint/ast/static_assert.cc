// Copyright 2022 The Tint Authors.
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

#include "src/tint/ast/static_assert.h"

#include "src/tint/program_builder.h"

TINT_INSTANTIATE_TYPEINFO(tint::ast::StaticAssert);

namespace tint::ast {

StaticAssert::StaticAssert(ProgramID pid, NodeID nid, const Source& src, const Expression* cond)
    : Base(pid, nid, src), condition(cond) {
    TINT_ASSERT(AST, cond);
    TINT_ASSERT_PROGRAM_IDS_EQUAL_IF_VALID(AST, cond, program_id);
}

StaticAssert::StaticAssert(StaticAssert&&) = default;

StaticAssert::~StaticAssert() = default;

const StaticAssert* StaticAssert::Clone(CloneContext* ctx) const {
    // Clone arguments outside of create() call to have deterministic ordering
    auto src = ctx->Clone(source);
    auto* cond = ctx->Clone(condition);
    return ctx->dst->create<StaticAssert>(src, cond);
}

}  // namespace tint::ast
