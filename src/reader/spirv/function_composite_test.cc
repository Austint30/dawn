// Copyright 2020 The Tint Authors.
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

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "src/reader/spirv/fail_stream.h"
#include "src/reader/spirv/function.h"
#include "src/reader/spirv/parser_impl.h"
#include "src/reader/spirv/parser_impl_test_helper.h"
#include "src/reader/spirv/spirv_tools_helpers_test.h"

namespace tint {
namespace reader {
namespace spirv {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

std::string Preamble() {
  return R"(
  %void = OpTypeVoid
  %voidfn = OpTypeFunction %void

  %uint = OpTypeInt 32 0
  %int = OpTypeInt 32 1
  %float = OpTypeFloat 32

  %uint_10 = OpConstant %uint 10
  %uint_20 = OpConstant %uint 20
  %uint_3 = OpConstant %uint 3
  %uint_4 = OpConstant %uint 4
  %uint_5 = OpConstant %uint 5
  %int_30 = OpConstant %int 30
  %int_40 = OpConstant %int 40
  %float_50 = OpConstant %float 50
  %float_60 = OpConstant %float 60
  %float_70 = OpConstant %float 70

  %v2uint = OpTypeVector %uint 2
  %v2int = OpTypeVector %int 2
  %v2float = OpTypeVector %float 2

  %m3v2float = OpTypeMatrix %v2float 3
  %m3v2float_0 = OpConstantNull %m3v2float

  %s_v2f_u_i = OpTypeStruct %v2float %uint %int
  %a_u_5 = OpTypeArray %uint %uint_5

  %v2float_50_60 = OpConstantComposite %v2float %float_50 %float_60
  %v2float_60_50 = OpConstantComposite %v2float %float_60 %float_50
  %v2float_70_70 = OpConstantComposite %v2float %float_70 %float_70
)";
}

using SpvParserTest_Composite_Construct = SpvParserTest;

TEST_F(SpvParserTest_Composite_Construct, Vector) {
  const auto assembly = Preamble() + R"(
     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpCompositeConstruct %v2uint %uint_10 %uint_20
     %2 = OpCompositeConstruct %v2int %int_30 %int_40
     %3 = OpCompositeConstruct %v2float %float_50 %float_60
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(VariableDeclStatement{
  Variable{
    x_1
    none
    __vec_2__u32
    {
      TypeConstructor{
        __vec_2__u32
        ScalarConstructor{10}
        ScalarConstructor{20}
      }
    }
  }
}
VariableDeclStatement{
  Variable{
    x_2
    none
    __vec_2__i32
    {
      TypeConstructor{
        __vec_2__i32
        ScalarConstructor{30}
        ScalarConstructor{40}
      }
    }
  }
}
VariableDeclStatement{
  Variable{
    x_3
    none
    __vec_2__f32
    {
      TypeConstructor{
        __vec_2__f32
        ScalarConstructor{50.000000}
        ScalarConstructor{60.000000}
      }
    }
  }
})")) << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_Composite_Construct, Matrix) {
  const auto assembly = Preamble() + R"(
     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpCompositeConstruct %m3v2float %v2float_50_60 %v2float_60_50 %v2float_70_70
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_1
    none
    __mat_2_3__f32
    {
      TypeConstructor{
        __mat_2_3__f32
        TypeConstructor{
          __vec_2__f32
          ScalarConstructor{50.000000}
          ScalarConstructor{60.000000}
        }
        TypeConstructor{
          __vec_2__f32
          ScalarConstructor{60.000000}
          ScalarConstructor{50.000000}
        }
        TypeConstructor{
          __vec_2__f32
          ScalarConstructor{70.000000}
          ScalarConstructor{70.000000}
        }
      }
    }
  })"))
      << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_Composite_Construct, Array) {
  const auto assembly = Preamble() + R"(
     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpCompositeConstruct %a_u_5 %uint_10 %uint_20 %uint_3 %uint_4 %uint_5
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_1
    none
    __array__u32_5
    {
      TypeConstructor{
        __array__u32_5
        ScalarConstructor{10}
        ScalarConstructor{20}
        ScalarConstructor{3}
        ScalarConstructor{4}
        ScalarConstructor{5}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_Composite_Construct, Struct) {
  const auto assembly = Preamble() + R"(
     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpCompositeConstruct %s_v2f_u_i %v2float_50_60 %uint_5 %int_30
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_1
    none
    __struct_S
    {
      TypeConstructor{
        __struct_S
        TypeConstructor{
          __vec_2__f32
          ScalarConstructor{50.000000}
          ScalarConstructor{60.000000}
        }
        ScalarConstructor{5}
        ScalarConstructor{30}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

using SpvParserTest_CompositeExtract = SpvParserTest;

TEST_F(SpvParserTest_CompositeExtract, Vector) {
  const auto assembly = Preamble() + R"(
     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpCompositeExtract %float %v2float_50_60 1
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_1
    none
    __f32
    {
      MemberAccessor{
        TypeConstructor{
          __vec_2__f32
          ScalarConstructor{50.000000}
          ScalarConstructor{60.000000}
        }
        Identifier{y}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_CompositeExtract, Vector_IndexTooBigError) {
  const auto assembly = Preamble() + R"(
     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpCompositeExtract %float %v2float_50_60 900
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_FALSE(fe.EmitBody());
  EXPECT_THAT(p->error(), Eq("CompositeExtract %1 index value 900 is out of "
                             "bounds for vector of 2 elements"));
}

TEST_F(SpvParserTest_CompositeExtract, Matrix) {
  const auto assembly = Preamble() + R"(
     %ptr = OpTypePointer Function %m3v2float
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %m3v2float %var
     %2 = OpCompositeExtract %v2float %1 2
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_2
    none
    __vec_2__f32
    {
      ArrayAccessor{
        Identifier{x_1}
        ScalarConstructor{2}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_CompositeExtract, Matrix_IndexTooBigError) {
  const auto assembly = Preamble() + R"(
     %ptr = OpTypePointer Function %m3v2float
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %m3v2float %var
     %2 = OpCompositeExtract %v2float %1 3
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_FALSE(fe.EmitBody()) << p->error();
  EXPECT_THAT(p->error(), Eq("CompositeExtract %2 index value 3 is out of "
                             "bounds for matrix of 3 elements"));
}

TEST_F(SpvParserTest_CompositeExtract, Matrix_Vector) {
  const auto assembly = Preamble() + R"(
     %ptr = OpTypePointer Function %m3v2float
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %m3v2float %var
     %2 = OpCompositeExtract %float %1 2 1
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_2
    none
    __f32
    {
      MemberAccessor{
        ArrayAccessor{
          Identifier{x_1}
          ScalarConstructor{2}
        }
        Identifier{y}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_CompositeExtract, Array) {
  const auto assembly = Preamble() + R"(
     %ptr = OpTypePointer Function %a_u_5
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %a_u_5 %var
     %2 = OpCompositeExtract %uint %1 3
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_2
    none
    __u32
    {
      ArrayAccessor{
        Identifier{x_1}
        ScalarConstructor{3}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_CompositeExtract, RuntimeArray_IsError) {
  const auto assembly = Preamble() + R"(
     %rtarr = OpTypeRuntimeArray %uint
     %ptr = OpTypePointer Function %rtarr
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %rtarr %var
     %2 = OpCompositeExtract %uint %1 3
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_FALSE(fe.EmitBody()) << p->error();
  EXPECT_THAT(p->error(), Eq("can't do OpCompositeExtract on a runtime array"));
}

TEST_F(SpvParserTest_CompositeExtract, Struct) {
  const auto assembly = Preamble() + R"(
     %ptr = OpTypePointer Function %s_v2f_u_i
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %s_v2f_u_i %var
     %2 = OpCompositeExtract %int %1 2
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_2
    none
    __i32
    {
      MemberAccessor{
        Identifier{x_1}
        Identifier{field2}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

TEST_F(SpvParserTest_CompositeExtract, Struct_IndexTooBigError) {
  const auto assembly = Preamble() + R"(
     %ptr = OpTypePointer Function %s_v2f_u_i
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %s_v2f_u_i %var
     %2 = OpCompositeExtract %int %1 40
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_FALSE(fe.EmitBody());
  EXPECT_THAT(p->error(), Eq("CompositeExtract %2 index value 40 is out of "
                             "bounds for structure %23 having 3 elements"));
}

TEST_F(SpvParserTest_CompositeExtract, Struct_Array_Matrix_Vector) {
  const auto assembly = Preamble() + R"(
     %a_mat = OpTypeArray %m3v2float %uint_3
     %s = OpTypeStruct %uint %a_mat
     %ptr = OpTypePointer Function %s
     %var = OpVariable %ptr Function

     %100 = OpFunction %void None %voidfn
     %entry = OpLabel
     %1 = OpLoad %s %var
     %2 = OpCompositeExtract %float %1 1 2 0 1
     OpReturn
     OpFunctionEnd
  )";
  auto* p = parser(test::Assemble(assembly));
  ASSERT_TRUE(p->BuildAndParseInternalModuleExceptFunctions()) << assembly;
  FunctionEmitter fe(p, *spirv_function(100));
  EXPECT_TRUE(fe.EmitBody()) << p->error();
  EXPECT_THAT(ToString(fe.ast_body()), HasSubstr(R"(
  Variable{
    x_2
    none
    __f32
    {
      MemberAccessor{
        ArrayAccessor{
          ArrayAccessor{
            MemberAccessor{
              Identifier{x_1}
              Identifier{field1}
            }
            ScalarConstructor{2}
          }
          ScalarConstructor{0}
        }
        Identifier{y}
      }
    }
  })"))
      << ToString(fe.ast_body());
}

}  // namespace
}  // namespace spirv
}  // namespace reader
}  // namespace tint
