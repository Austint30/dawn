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

#ifndef SRC_TINT_WRITER_HLSL_GENERATOR_IMPL_H_
#define SRC_TINT_WRITER_HLSL_GENERATOR_IMPL_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "src/tint/ast/assignment_statement.h"
#include "src/tint/ast/bitcast_expression.h"
#include "src/tint/ast/break_statement.h"
#include "src/tint/ast/continue_statement.h"
#include "src/tint/ast/discard_statement.h"
#include "src/tint/ast/for_loop_statement.h"
#include "src/tint/ast/if_statement.h"
#include "src/tint/ast/loop_statement.h"
#include "src/tint/ast/return_statement.h"
#include "src/tint/ast/switch_statement.h"
#include "src/tint/ast/unary_op_expression.h"
#include "src/tint/program_builder.h"
#include "src/tint/scope_stack.h"
#include "src/tint/sem/binding_point.h"
#include "src/tint/transform/decompose_memory_access.h"
#include "src/tint/utils/hash.h"
#include "src/tint/writer/array_length_from_uniform_options.h"
#include "src/tint/writer/hlsl/generator.h"
#include "src/tint/writer/text_generator.h"

// Forward declarations
namespace tint::sem {
class Call;
class Constant;
class Builtin;
class TypeConstructor;
class TypeConversion;
}  // namespace tint::sem

namespace tint::writer::hlsl {

/// The result of sanitizing a program for generation.
struct SanitizedResult {
    /// Constructor
    SanitizedResult();
    /// Destructor
    ~SanitizedResult();
    /// Move constructor
    SanitizedResult(SanitizedResult&&);

    /// The sanitized program.
    Program program;
    /// Indices into the array_length_from_uniform binding that are statically
    /// used.
    std::unordered_set<uint32_t> used_array_length_from_uniform_indices;
};

/// Sanitize a program in preparation for generating HLSL.
/// @param program the input program
/// @param options The HLSL generator options.
/// @returns the sanitized program and any supplementary information
SanitizedResult Sanitize(const Program* program, const Options& options);

/// Implementation class for HLSL generator
class GeneratorImpl : public TextGenerator {
  public:
    /// Constructor
    /// @param program the program to generate
    explicit GeneratorImpl(const Program* program);
    ~GeneratorImpl();

    /// @returns true on successful generation; false otherwise
    bool Generate();

    /// Handles an index accessor expression
    /// @param out the output of the expression stream
    /// @param expr the expression to emit
    /// @returns true if the index accessor was emitted
    bool EmitIndexAccessor(std::ostream& out, const ast::IndexAccessorExpression* expr);
    /// Handles an assignment statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted successfully
    bool EmitAssign(const ast::AssignmentStatement* stmt);
    /// Emits code such that if `expr` is zero, it emits one, else `expr`
    /// @param out the output of the expression stream
    /// @param expr the expression
    /// @returns true if the expression was emitted, false otherwise
    bool EmitExpressionOrOneIfZero(std::ostream& out, const ast::Expression* expr);
    /// Handles generating a binary expression
    /// @param out the output of the expression stream
    /// @param expr the binary expression
    /// @returns true if the expression was emitted, false otherwise
    bool EmitBinary(std::ostream& out, const ast::BinaryExpression* expr);
    /// Handles generating a bitcast expression
    /// @param out the output of the expression stream
    /// @param expr the as expression
    /// @returns true if the bitcast was emitted
    bool EmitBitcast(std::ostream& out, const ast::BitcastExpression* expr);
    /// Emits a list of statements
    /// @param stmts the statement list
    /// @returns true if the statements were emitted successfully
    bool EmitStatements(const ast::StatementList& stmts);
    /// Emits a list of statements with an indentation
    /// @param stmts the statement list
    /// @returns true if the statements were emitted successfully
    bool EmitStatementsWithIndent(const ast::StatementList& stmts);
    /// Handles a block statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted successfully
    bool EmitBlock(const ast::BlockStatement* stmt);
    /// Handles a break statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted successfully
    bool EmitBreak(const ast::BreakStatement* stmt);
    /// Handles generating a call expression
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @returns true if the call expression is emitted
    bool EmitCall(std::ostream& out, const ast::CallExpression* expr);
    /// Handles generating a function call expression
    /// @param out the output of the expression stream
    /// @param call the call expression
    /// @param function the function being called
    /// @returns true if the expression is emitted
    bool EmitFunctionCall(std::ostream& out, const sem::Call* call, const sem::Function* function);
    /// Handles generating a builtin call expression
    /// @param out the output of the expression stream
    /// @param call the call expression
    /// @param builtin the builtin being called
    /// @returns true if the expression is emitted
    bool EmitBuiltinCall(std::ostream& out, const sem::Call* call, const sem::Builtin* builtin);
    /// Handles generating a type conversion expression
    /// @param out the output of the expression stream
    /// @param call the call expression
    /// @param conv the type conversion
    /// @returns true if the expression is emitted
    bool EmitTypeConversion(std::ostream& out,
                            const sem::Call* call,
                            const sem::TypeConversion* conv);
    /// Handles generating a type constructor expression
    /// @param out the output of the expression stream
    /// @param call the call expression
    /// @param ctor the type constructor
    /// @returns true if the expression is emitted
    bool EmitTypeConstructor(std::ostream& out,
                             const sem::Call* call,
                             const sem::TypeConstructor* ctor);
    /// Handles generating a call expression to a
    /// transform::DecomposeMemoryAccess::Intrinsic for a uniform buffer
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param intrinsic the transform::DecomposeMemoryAccess::Intrinsic
    /// @returns true if the call expression is emitted
    bool EmitUniformBufferAccess(std::ostream& out,
                                 const ast::CallExpression* expr,
                                 const transform::DecomposeMemoryAccess::Intrinsic* intrinsic);
    /// Handles generating a call expression to a
    /// transform::DecomposeMemoryAccess::Intrinsic for a storage buffer
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param intrinsic the transform::DecomposeMemoryAccess::Intrinsic
    /// @returns true if the call expression is emitted
    bool EmitStorageBufferAccess(std::ostream& out,
                                 const ast::CallExpression* expr,
                                 const transform::DecomposeMemoryAccess::Intrinsic* intrinsic);
    /// Handles generating a barrier intrinsic call
    /// @param out the output of the expression stream
    /// @param builtin the semantic information for the barrier builtin
    /// @returns true if the call expression is emitted
    bool EmitBarrierCall(std::ostream& out, const sem::Builtin* builtin);
    /// Handles generating an atomic intrinsic call for a storage buffer variable
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param intrinsic the atomic intrinsic
    /// @returns true if the call expression is emitted
    bool EmitStorageAtomicCall(std::ostream& out,
                               const ast::CallExpression* expr,
                               const transform::DecomposeMemoryAccess::Intrinsic* intrinsic);
    /// Handles generating the helper function for the atomic intrinsic function
    /// @param func the function
    /// @param intrinsic the atomic intrinsic
    /// @returns true if the function is emitted
    bool EmitStorageAtomicIntrinsic(const ast::Function* func,
                                    const transform::DecomposeMemoryAccess::Intrinsic* intrinsic);
    /// Handles generating an atomic intrinsic call for a workgroup variable
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the atomic builtin
    /// @returns true if the call expression is emitted
    bool EmitWorkgroupAtomicCall(std::ostream& out,
                                 const ast::CallExpression* expr,
                                 const sem::Builtin* builtin);
    /// Handles generating a call to a texture function (`textureSample`,
    /// `textureSampleGrad`, etc)
    /// @param out the output of the expression stream
    /// @param call the call expression
    /// @param builtin the semantic information for the texture builtin
    /// @returns true if the call expression is emitted
    bool EmitTextureCall(std::ostream& out, const sem::Call* call, const sem::Builtin* builtin);
    /// Handles generating a call to the `select()` builtin
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @returns true if the call expression is emitted
    bool EmitSelectCall(std::ostream& out, const ast::CallExpression* expr);
    /// Handles generating a call to the `modf()` builtin
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the builtin
    /// @returns true if the call expression is emitted
    bool EmitModfCall(std::ostream& out,
                      const ast::CallExpression* expr,
                      const sem::Builtin* builtin);
    /// Handles generating a call to the `frexp()` builtin
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the builtin
    /// @returns true if the call expression is emitted
    bool EmitFrexpCall(std::ostream& out,
                       const ast::CallExpression* expr,
                       const sem::Builtin* builtin);
    /// Handles generating a call to the `degrees()` builtin
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the builtin
    /// @returns true if the call expression is emitted
    bool EmitDegreesCall(std::ostream& out,
                         const ast::CallExpression* expr,
                         const sem::Builtin* builtin);
    /// Handles generating a call to the `radians()` builtin
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the builtin
    /// @returns true if the call expression is emitted
    bool EmitRadiansCall(std::ostream& out,
                         const ast::CallExpression* expr,
                         const sem::Builtin* builtin);
    /// Handles generating a call to data packing builtin
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the builtin
    /// @returns true if the call expression is emitted
    bool EmitDataPackingCall(std::ostream& out,
                             const ast::CallExpression* expr,
                             const sem::Builtin* builtin);
    /// Handles generating a call to data unpacking builtin
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the builtin
    /// @returns true if the call expression is emitted
    bool EmitDataUnpackingCall(std::ostream& out,
                               const ast::CallExpression* expr,
                               const sem::Builtin* builtin);
    /// Handles generating a call to DP4a builtins (dot4I8Packed and dot4U8Packed)
    /// @param out the output of the expression stream
    /// @param expr the call expression
    /// @param builtin the semantic information for the builtin
    /// @returns true if the call expression is emitted
    bool EmitDP4aCall(std::ostream& out,
                      const ast::CallExpression* expr,
                      const sem::Builtin* builtin);
    /// Handles a case statement
    /// @param s the switch statement
    /// @param case_idx the index of the switch case in the switch statement
    /// @returns true if the statement was emitted successfully
    bool EmitCase(const ast::SwitchStatement* s, size_t case_idx);
    /// Handles generating a discard statement
    /// @param stmt the discard statement
    /// @returns true if the statement was successfully emitted
    bool EmitDiscard(const ast::DiscardStatement* stmt);
    /// Handles a continue statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted successfully
    bool EmitContinue(const ast::ContinueStatement* stmt);
    /// Handles generate an Expression
    /// @param out the output of the expression stream
    /// @param expr the expression
    /// @returns true if the expression was emitted
    bool EmitExpression(std::ostream& out, const ast::Expression* expr);
    /// Handles generating a function
    /// @param func the function to generate
    /// @returns true if the function was emitted
    bool EmitFunction(const ast::Function* func);
    /// Handles emitting the function body if it discards to work around a FXC
    /// compilation bug.
    /// @param func the function with the body to emit
    /// @returns true if the function was emitted
    bool EmitFunctionBodyWithDiscard(const ast::Function* func);
    /// Handles emitting a global variable
    /// @param global the global variable
    /// @returns true on success
    bool EmitGlobalVariable(const ast::Variable* global);

    /// Handles emitting a global variable with the uniform storage class
    /// @param var the AST node for the 'var'
    /// @param sem the semantic node for the 'var'
    /// @returns true on success
    bool EmitUniformVariable(const ast::Var* var, const sem::Variable* sem);

    /// Handles emitting a global variable with the storage storage class
    /// @param var the AST node for the 'var'
    /// @param sem the semantic node for the 'var'
    /// @returns true on success
    bool EmitStorageVariable(const ast::Var* var, const sem::Variable* sem);

    /// Handles emitting a global variable with the handle storage class
    /// @param var the AST node for the 'var'
    /// @param sem the semantic node for the 'var'
    /// @returns true on success
    bool EmitHandleVariable(const ast::Var* var, const sem::Variable* sem);

    /// Handles emitting a global variable with the private storage class
    /// @param var the global variable
    /// @returns true on success
    bool EmitPrivateVariable(const sem::Variable* var);

    /// Handles emitting a global variable with the workgroup storage class
    /// @param var the global variable
    /// @returns true on success
    bool EmitWorkgroupVariable(const sem::Variable* var);

    /// Handles emitting the entry point function
    /// @param func the entry point
    /// @returns true if the entry point function was emitted
    bool EmitEntryPointFunction(const ast::Function* func);
    /// Handles an if statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was successfully emitted
    bool EmitIf(const ast::IfStatement* stmt);
    /// Handles a constant value
    /// @param out the output stream
    /// @param constant the constant value to emit
    /// @returns true if the constant value was successfully emitted
    bool EmitConstant(std::ostream& out, const sem::Constant& constant);
    /// Handles emitting a sub-range of a constant value
    /// @param out the output stream
    /// @param constant the constant value to emit
    /// @param range_ty the sub-range type
    /// @param start the element index for the first element
    /// @param end the element index for one past the last element
    /// @returns true if the constant value was successfully emitted
    bool EmitConstantRange(std::ostream& out,
                           const sem::Constant& constant,
                           const sem::Type* range_ty,
                           size_t start,
                           size_t end);
    /// Handles a literal
    /// @param out the output stream
    /// @param lit the literal to emit
    /// @returns true if the literal was successfully emitted
    bool EmitLiteral(std::ostream& out, const ast::LiteralExpression* lit);
    /// Handles a loop statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted
    bool EmitLoop(const ast::LoopStatement* stmt);
    /// Handles a for loop statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted
    bool EmitForLoop(const ast::ForLoopStatement* stmt);
    /// Handles a while statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted
    bool EmitWhile(const ast::WhileStatement* stmt);
    /// Handles generating an identifier expression
    /// @param out the output of the expression stream
    /// @param expr the identifier expression
    /// @returns true if the identifeir was emitted
    bool EmitIdentifier(std::ostream& out, const ast::IdentifierExpression* expr);
    /// Handles a member accessor expression
    /// @param out the output of the expression stream
    /// @param expr the member accessor expression
    /// @returns true if the member accessor was emitted
    bool EmitMemberAccessor(std::ostream& out, const ast::MemberAccessorExpression* expr);
    /// Handles return statements
    /// @param stmt the statement to emit
    /// @returns true if the statement was successfully emitted
    bool EmitReturn(const ast::ReturnStatement* stmt);
    /// Handles statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted
    bool EmitStatement(const ast::Statement* stmt);
    /// Handles generating a switch statement
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted
    bool EmitSwitch(const ast::SwitchStatement* stmt);
    // Handles generating a switch statement with only a default case
    /// @param stmt the statement to emit
    /// @returns true if the statement was emitted
    bool EmitDefaultOnlySwitch(const ast::SwitchStatement* stmt);
    /// Handles generating type
    /// @param out the output stream
    /// @param type the type to generate
    /// @param storage_class the storage class of the variable
    /// @param access the access control type of the variable
    /// @param name the name of the variable, used for array emission.
    /// @param name_printed (optional) if not nullptr and an array was printed
    /// then the boolean is set to true.
    /// @returns true if the type is emitted
    bool EmitType(std::ostream& out,
                  const sem::Type* type,
                  ast::StorageClass storage_class,
                  ast::Access access,
                  const std::string& name,
                  bool* name_printed = nullptr);
    /// Handles generating type and name
    /// @param out the output stream
    /// @param type the type to generate
    /// @param storage_class the storage class of the variable
    /// @param access the access control type of the variable
    /// @param name the name to emit
    /// @returns true if the type is emitted
    bool EmitTypeAndName(std::ostream& out,
                         const sem::Type* type,
                         ast::StorageClass storage_class,
                         ast::Access access,
                         const std::string& name);
    /// Handles generating a structure declaration
    /// @param buffer the text buffer that the type declaration will be written to
    /// @param ty the struct to generate
    /// @returns true if the struct is emitted
    bool EmitStructType(TextBuffer* buffer, const sem::Struct* ty);
    /// Handles generating a structure declaration only the first time called. Subsequent calls are
    /// a no-op and return true.
    /// @param buffer the text buffer that the type declaration will be written to
    /// @param ty the struct to generate
    /// @returns true if the struct is emitted
    bool EmitStructTypeOnce(TextBuffer* buffer, const sem::Struct* ty);
    /// Handles a unary op expression
    /// @param out the output of the expression stream
    /// @param expr the expression to emit
    /// @returns true if the expression was emitted
    bool EmitUnaryOp(std::ostream& out, const ast::UnaryOpExpression* expr);
    /// Emits `value` for the given type
    /// @param out the output stream
    /// @param type the type to emit the value for
    /// @param value the value to emit
    /// @returns true if the value was successfully emitted.
    bool EmitValue(std::ostream& out, const sem::Type* type, int value);
    /// Emits the zero value for the given type
    /// @param out the output stream
    /// @param type the type to emit the value for
    /// @returns true if the zero value was successfully emitted.
    bool EmitZeroValue(std::ostream& out, const sem::Type* type);
    /// Handles generating a 'var' declaration
    /// @param var the variable to generate
    /// @returns true if the variable was emitted
    bool EmitVar(const ast::Var* var);
    /// Handles generating a function-scope 'let' declaration
    /// @param let the variable to generate
    /// @returns true if the variable was emitted
    bool EmitLet(const ast::Let* let);
    /// Handles generating a module-scope 'override' declaration
    /// @param override the 'override' to emit
    /// @returns true if the variable was emitted
    bool EmitOverride(const ast::Override* override);
    /// Emits call to a helper vector assignment function for the input assignment
    /// statement and vector type. This is used to work around FXC issues where
    /// assignments to vectors with dynamic indices cause compilation failures.
    /// @param stmt assignment statement that corresponds to a vector assignment
    /// via an accessor expression
    /// @param vec the vector type being assigned to
    /// @returns true on success
    bool EmitDynamicVectorAssignment(const ast::AssignmentStatement* stmt, const sem::Vector* vec);
    /// Emits call to a helper matrix assignment function for the input assignment
    /// statement and matrix type. This is used to work around FXC issues where
    /// assignment of a vector to a matrix with a dynamic index causes compilation
    /// failures.
    /// @param stmt assignment statement that corresponds to a matrix assignment
    /// via an accessor expression
    /// @param mat the matrix type being assigned to
    /// @returns true on success
    bool EmitDynamicMatrixVectorAssignment(const ast::AssignmentStatement* stmt,
                                           const sem::Matrix* mat);
    /// Emits call to a helper matrix assignment function for the input assignment
    /// statement and matrix type. This is used to work around FXC issues where
    /// assignment of a scalar to a matrix with at least one dynamic index causes
    /// compilation failures.
    /// @param stmt assignment statement that corresponds to a matrix assignment
    /// via an accessor expression
    /// @param mat the matrix type being assigned to
    /// @returns true on success
    bool EmitDynamicMatrixScalarAssignment(const ast::AssignmentStatement* stmt,
                                           const sem::Matrix* mat);

    /// Handles generating a builtin method name
    /// @param builtin the semantic info for the builtin
    /// @returns the name or "" if not valid
    std::string generate_builtin_name(const sem::Builtin* builtin);
    /// Converts a builtin to an attribute name
    /// @param builtin the builtin to convert
    /// @returns the string name of the builtin or blank on error
    std::string builtin_to_attribute(ast::Builtin builtin) const;

    /// Converts interpolation attributes to a HLSL modifiers
    /// @param type the interpolation type
    /// @param sampling the interpolation sampling
    /// @returns the string name of the attribute or blank on error
    std::string interpolation_to_modifiers(ast::InterpolationType type,
                                           ast::InterpolationSampling sampling) const;

  private:
    enum class VarType { kIn, kOut };

    struct EntryPointData {
        std::string struct_name;
        std::string var_name;
    };

    struct DMAIntrinsic {
        transform::DecomposeMemoryAccess::Intrinsic::Op op;
        transform::DecomposeMemoryAccess::Intrinsic::DataType type;
        bool operator==(const DMAIntrinsic& rhs) const { return op == rhs.op && type == rhs.type; }
        /// Hasher is a std::hash function for DMAIntrinsic
        struct Hasher {
            /// @param i the DMAIntrinsic to hash
            /// @returns the hash of `i`
            inline std::size_t operator()(const DMAIntrinsic& i) const {
                return utils::Hash(i.op, i.type);
            }
        };
    };

    /// CallBuiltinHelper will call the builtin helper function, creating it
    /// if it hasn't been built already. If the builtin needs to be built then
    /// CallBuiltinHelper will generate the function signature and will call
    /// `build` to emit the body of the function.
    /// @param out the output of the expression stream
    /// @param call the call expression
    /// @param builtin the semantic information for the builtin
    /// @param build a function with the signature:
    ///        `bool(TextBuffer* buffer, const std::vector<std::string>& params)`
    ///        Where:
    ///          `buffer` is the body of the generated function
    ///          `params` is the name of all the generated function parameters
    /// @returns true if the call expression is emitted
    template <typename F>
    bool CallBuiltinHelper(std::ostream& out,
                           const ast::CallExpression* call,
                           const sem::Builtin* builtin,
                           F&& build);

    TextBuffer helpers_;  // Helper functions emitted at the top of the output
    std::function<bool()> emit_continuing_;
    std::unordered_map<const sem::Matrix*, std::string> matrix_scalar_ctors_;
    std::unordered_map<const sem::Builtin*, std::string> builtins_;
    std::unordered_map<const sem::Struct*, std::string> structure_builders_;
    std::unordered_map<const sem::Vector*, std::string> dynamic_vector_write_;
    std::unordered_map<const sem::Matrix*, std::string> dynamic_matrix_vector_write_;
    std::unordered_map<const sem::Matrix*, std::string> dynamic_matrix_scalar_write_;
    std::unordered_map<const sem::Type*, std::string> value_or_one_if_zero_;
    std::unordered_set<const sem::Struct*> emitted_structs_;
};

}  // namespace tint::writer::hlsl

#endif  // SRC_TINT_WRITER_HLSL_GENERATOR_IMPL_H_
