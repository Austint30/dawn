# Copyright 2023 The Dawn & Tint Authors
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################
# File generated by 'tools/src/cmd/gen' using the template:
#   tools/src/cmd/gen/build/BUILD.cmake.tmpl
#
# To regenerate run: './tools/run gen'
#
#                       Do not modify this file directly
################################################################################

include(lang/wgsl/ast/transform/BUILD.cmake)

################################################################################
# Target:    tint_lang_wgsl_ast
# Kind:      lib
################################################################################
tint_add_target(tint_lang_wgsl_ast lib
  lang/wgsl/ast/accessor_expression.cc
  lang/wgsl/ast/accessor_expression.h
  lang/wgsl/ast/alias.cc
  lang/wgsl/ast/alias.h
  lang/wgsl/ast/assignment_statement.cc
  lang/wgsl/ast/assignment_statement.h
  lang/wgsl/ast/attribute.cc
  lang/wgsl/ast/attribute.h
  lang/wgsl/ast/binary_expression.cc
  lang/wgsl/ast/binary_expression.h
  lang/wgsl/ast/binding_attribute.cc
  lang/wgsl/ast/binding_attribute.h
  lang/wgsl/ast/bitcast_expression.cc
  lang/wgsl/ast/bitcast_expression.h
  lang/wgsl/ast/block_statement.cc
  lang/wgsl/ast/block_statement.h
  lang/wgsl/ast/bool_literal_expression.cc
  lang/wgsl/ast/bool_literal_expression.h
  lang/wgsl/ast/break_if_statement.cc
  lang/wgsl/ast/break_if_statement.h
  lang/wgsl/ast/break_statement.cc
  lang/wgsl/ast/break_statement.h
  lang/wgsl/ast/builder.cc
  lang/wgsl/ast/builder.h
  lang/wgsl/ast/builtin_attribute.cc
  lang/wgsl/ast/builtin_attribute.h
  lang/wgsl/ast/call_expression.cc
  lang/wgsl/ast/call_expression.h
  lang/wgsl/ast/call_statement.cc
  lang/wgsl/ast/call_statement.h
  lang/wgsl/ast/case_selector.cc
  lang/wgsl/ast/case_selector.h
  lang/wgsl/ast/case_statement.cc
  lang/wgsl/ast/case_statement.h
  lang/wgsl/ast/clone_context.cc
  lang/wgsl/ast/clone_context.h
  lang/wgsl/ast/color_attribute.cc
  lang/wgsl/ast/color_attribute.h
  lang/wgsl/ast/compound_assignment_statement.cc
  lang/wgsl/ast/compound_assignment_statement.h
  lang/wgsl/ast/const.cc
  lang/wgsl/ast/const.h
  lang/wgsl/ast/const_assert.cc
  lang/wgsl/ast/const_assert.h
  lang/wgsl/ast/continue_statement.cc
  lang/wgsl/ast/continue_statement.h
  lang/wgsl/ast/diagnostic_attribute.cc
  lang/wgsl/ast/diagnostic_attribute.h
  lang/wgsl/ast/diagnostic_control.cc
  lang/wgsl/ast/diagnostic_control.h
  lang/wgsl/ast/diagnostic_directive.cc
  lang/wgsl/ast/diagnostic_directive.h
  lang/wgsl/ast/diagnostic_rule_name.cc
  lang/wgsl/ast/diagnostic_rule_name.h
  lang/wgsl/ast/disable_validation_attribute.cc
  lang/wgsl/ast/disable_validation_attribute.h
  lang/wgsl/ast/discard_statement.cc
  lang/wgsl/ast/discard_statement.h
  lang/wgsl/ast/enable.cc
  lang/wgsl/ast/enable.h
  lang/wgsl/ast/expression.cc
  lang/wgsl/ast/expression.h
  lang/wgsl/ast/extension.cc
  lang/wgsl/ast/extension.h
  lang/wgsl/ast/float_literal_expression.cc
  lang/wgsl/ast/float_literal_expression.h
  lang/wgsl/ast/for_loop_statement.cc
  lang/wgsl/ast/for_loop_statement.h
  lang/wgsl/ast/function.cc
  lang/wgsl/ast/function.h
  lang/wgsl/ast/group_attribute.cc
  lang/wgsl/ast/group_attribute.h
  lang/wgsl/ast/id_attribute.cc
  lang/wgsl/ast/id_attribute.h
  lang/wgsl/ast/identifier.cc
  lang/wgsl/ast/identifier.h
  lang/wgsl/ast/identifier_expression.cc
  lang/wgsl/ast/identifier_expression.h
  lang/wgsl/ast/if_statement.cc
  lang/wgsl/ast/if_statement.h
  lang/wgsl/ast/increment_decrement_statement.cc
  lang/wgsl/ast/increment_decrement_statement.h
  lang/wgsl/ast/index_accessor_expression.cc
  lang/wgsl/ast/index_accessor_expression.h
  lang/wgsl/ast/index_attribute.cc
  lang/wgsl/ast/index_attribute.h
  lang/wgsl/ast/int_literal_expression.cc
  lang/wgsl/ast/int_literal_expression.h
  lang/wgsl/ast/internal_attribute.cc
  lang/wgsl/ast/internal_attribute.h
  lang/wgsl/ast/interpolate_attribute.cc
  lang/wgsl/ast/interpolate_attribute.h
  lang/wgsl/ast/invariant_attribute.cc
  lang/wgsl/ast/invariant_attribute.h
  lang/wgsl/ast/let.cc
  lang/wgsl/ast/let.h
  lang/wgsl/ast/literal_expression.cc
  lang/wgsl/ast/literal_expression.h
  lang/wgsl/ast/location_attribute.cc
  lang/wgsl/ast/location_attribute.h
  lang/wgsl/ast/loop_statement.cc
  lang/wgsl/ast/loop_statement.h
  lang/wgsl/ast/member_accessor_expression.cc
  lang/wgsl/ast/member_accessor_expression.h
  lang/wgsl/ast/module.cc
  lang/wgsl/ast/module.h
  lang/wgsl/ast/must_use_attribute.cc
  lang/wgsl/ast/must_use_attribute.h
  lang/wgsl/ast/node.cc
  lang/wgsl/ast/node.h
  lang/wgsl/ast/node_id.h
  lang/wgsl/ast/override.cc
  lang/wgsl/ast/override.h
  lang/wgsl/ast/parameter.cc
  lang/wgsl/ast/parameter.h
  lang/wgsl/ast/phony_expression.cc
  lang/wgsl/ast/phony_expression.h
  lang/wgsl/ast/pipeline_stage.cc
  lang/wgsl/ast/pipeline_stage.h
  lang/wgsl/ast/requires.cc
  lang/wgsl/ast/requires.h
  lang/wgsl/ast/return_statement.cc
  lang/wgsl/ast/return_statement.h
  lang/wgsl/ast/stage_attribute.cc
  lang/wgsl/ast/stage_attribute.h
  lang/wgsl/ast/statement.cc
  lang/wgsl/ast/statement.h
  lang/wgsl/ast/stride_attribute.cc
  lang/wgsl/ast/stride_attribute.h
  lang/wgsl/ast/struct.cc
  lang/wgsl/ast/struct.h
  lang/wgsl/ast/struct_member.cc
  lang/wgsl/ast/struct_member.h
  lang/wgsl/ast/struct_member_align_attribute.cc
  lang/wgsl/ast/struct_member_align_attribute.h
  lang/wgsl/ast/struct_member_offset_attribute.cc
  lang/wgsl/ast/struct_member_offset_attribute.h
  lang/wgsl/ast/struct_member_size_attribute.cc
  lang/wgsl/ast/struct_member_size_attribute.h
  lang/wgsl/ast/switch_statement.cc
  lang/wgsl/ast/switch_statement.h
  lang/wgsl/ast/templated_identifier.cc
  lang/wgsl/ast/templated_identifier.h
  lang/wgsl/ast/traverse_expressions.h
  lang/wgsl/ast/type.cc
  lang/wgsl/ast/type.h
  lang/wgsl/ast/type_decl.cc
  lang/wgsl/ast/type_decl.h
  lang/wgsl/ast/unary_op_expression.cc
  lang/wgsl/ast/unary_op_expression.h
  lang/wgsl/ast/var.cc
  lang/wgsl/ast/var.h
  lang/wgsl/ast/variable.cc
  lang/wgsl/ast/variable.h
  lang/wgsl/ast/variable_decl_statement.cc
  lang/wgsl/ast/variable_decl_statement.h
  lang/wgsl/ast/while_statement.cc
  lang/wgsl/ast/while_statement.h
  lang/wgsl/ast/workgroup_attribute.cc
  lang/wgsl/ast/workgroup_attribute.h
)

tint_target_add_dependencies(tint_lang_wgsl_ast lib
  tint_api_common
  tint_lang_core
  tint_lang_core_constant
  tint_lang_core_type
  tint_lang_wgsl
  tint_lang_wgsl_features
  tint_utils_containers
  tint_utils_diagnostic
  tint_utils_ice
  tint_utils_id
  tint_utils_macros
  tint_utils_math
  tint_utils_memory
  tint_utils_reflection
  tint_utils_result
  tint_utils_rtti
  tint_utils_symbol
  tint_utils_text
  tint_utils_traits
)

################################################################################
# Target:    tint_lang_wgsl_ast_test
# Kind:      test
################################################################################
tint_add_target(tint_lang_wgsl_ast_test test
  lang/wgsl/ast/alias_test.cc
  lang/wgsl/ast/assignment_statement_test.cc
  lang/wgsl/ast/binary_expression_test.cc
  lang/wgsl/ast/binding_attribute_test.cc
  lang/wgsl/ast/bitcast_expression_test.cc
  lang/wgsl/ast/block_statement_test.cc
  lang/wgsl/ast/bool_literal_expression_test.cc
  lang/wgsl/ast/break_if_statement_test.cc
  lang/wgsl/ast/break_statement_test.cc
  lang/wgsl/ast/builtin_attribute_test.cc
  lang/wgsl/ast/builtin_texture_helper_test.cc
  lang/wgsl/ast/builtin_texture_helper_test.h
  lang/wgsl/ast/call_expression_test.cc
  lang/wgsl/ast/call_statement_test.cc
  lang/wgsl/ast/case_selector_test.cc
  lang/wgsl/ast/case_statement_test.cc
  lang/wgsl/ast/clone_context_test.cc
  lang/wgsl/ast/color_attribute_test.cc
  lang/wgsl/ast/compound_assignment_statement_test.cc
  lang/wgsl/ast/const_assert_test.cc
  lang/wgsl/ast/continue_statement_test.cc
  lang/wgsl/ast/diagnostic_attribute_test.cc
  lang/wgsl/ast/diagnostic_control_test.cc
  lang/wgsl/ast/diagnostic_directive_test.cc
  lang/wgsl/ast/diagnostic_rule_name_test.cc
  lang/wgsl/ast/discard_statement_test.cc
  lang/wgsl/ast/enable_test.cc
  lang/wgsl/ast/float_literal_expression_test.cc
  lang/wgsl/ast/for_loop_statement_test.cc
  lang/wgsl/ast/function_test.cc
  lang/wgsl/ast/group_attribute_test.cc
  lang/wgsl/ast/helper_test.cc
  lang/wgsl/ast/helper_test.h
  lang/wgsl/ast/id_attribute_test.cc
  lang/wgsl/ast/identifier_expression_test.cc
  lang/wgsl/ast/identifier_test.cc
  lang/wgsl/ast/if_statement_test.cc
  lang/wgsl/ast/increment_decrement_statement_test.cc
  lang/wgsl/ast/index_accessor_expression_test.cc
  lang/wgsl/ast/index_attribute_test.cc
  lang/wgsl/ast/int_literal_expression_test.cc
  lang/wgsl/ast/interpolate_attribute_test.cc
  lang/wgsl/ast/location_attribute_test.cc
  lang/wgsl/ast/loop_statement_test.cc
  lang/wgsl/ast/member_accessor_expression_test.cc
  lang/wgsl/ast/module_test.cc
  lang/wgsl/ast/phony_expression_test.cc
  lang/wgsl/ast/requires_test.cc
  lang/wgsl/ast/return_statement_test.cc
  lang/wgsl/ast/stage_attribute_test.cc
  lang/wgsl/ast/stride_attribute_test.cc
  lang/wgsl/ast/struct_member_align_attribute_test.cc
  lang/wgsl/ast/struct_member_offset_attribute_test.cc
  lang/wgsl/ast/struct_member_size_attribute_test.cc
  lang/wgsl/ast/struct_member_test.cc
  lang/wgsl/ast/struct_test.cc
  lang/wgsl/ast/switch_statement_test.cc
  lang/wgsl/ast/templated_identifier_test.cc
  lang/wgsl/ast/traverse_expressions_test.cc
  lang/wgsl/ast/unary_op_expression_test.cc
  lang/wgsl/ast/variable_decl_statement_test.cc
  lang/wgsl/ast/variable_test.cc
  lang/wgsl/ast/while_statement_test.cc
  lang/wgsl/ast/workgroup_attribute_test.cc
)

tint_target_add_dependencies(tint_lang_wgsl_ast_test test
  tint_api_common
  tint_lang_core
  tint_lang_core_constant
  tint_lang_core_ir
  tint_lang_core_type
  tint_lang_wgsl
  tint_lang_wgsl_ast
  tint_lang_wgsl_ast_transform
  tint_lang_wgsl_common
  tint_lang_wgsl_features
  tint_lang_wgsl_program
  tint_lang_wgsl_resolver
  tint_lang_wgsl_sem
  tint_lang_wgsl_writer_ir_to_program
  tint_utils_containers
  tint_utils_diagnostic
  tint_utils_ice
  tint_utils_id
  tint_utils_macros
  tint_utils_math
  tint_utils_memory
  tint_utils_reflection
  tint_utils_result
  tint_utils_rtti
  tint_utils_symbol
  tint_utils_text
  tint_utils_traits
)

tint_target_add_external_dependencies(tint_lang_wgsl_ast_test test
  "gtest"
)

if(TINT_BUILD_WGSL_READER)
  tint_target_add_dependencies(tint_lang_wgsl_ast_test test
    tint_lang_wgsl_reader
  )
endif(TINT_BUILD_WGSL_READER)

if(TINT_BUILD_WGSL_READER AND TINT_BUILD_WGSL_WRITER)
  tint_target_add_sources(tint_lang_wgsl_ast_test test
    "lang/wgsl/ast/module_clone_test.cc"
  )
endif(TINT_BUILD_WGSL_READER AND TINT_BUILD_WGSL_WRITER)

if(TINT_BUILD_WGSL_WRITER)
  tint_target_add_dependencies(tint_lang_wgsl_ast_test test
    tint_lang_wgsl_writer
  )
endif(TINT_BUILD_WGSL_WRITER)
