// Copyright 2023 The Dawn & Tint Authors
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "src/tint/utils/bytes/reader.h"

#include "gtest/gtest.h"

namespace tint::bytes {
namespace {

template <typename... ARGS>
auto Data(ARGS&&... args) {
    return std::array{std::byte{static_cast<uint8_t>(args)}...};
}

TEST(BufferReaderTest, IntegerBigEndian) {
    auto data = Data(0x10, 0x20, 0x30, 0x40);
    auto u32 = BufferReader{Slice{data}}.Int<uint32_t>(Endianness::kBig);
    EXPECT_EQ(u32, 0x10203040u);
    auto i32 = BufferReader{Slice{data}}.Int<int32_t>(Endianness::kBig);
    EXPECT_EQ(i32, 0x10203040);
}

TEST(BufferReaderTest, IntegerBigEndian_TooShort) {
    auto data = Data(0x10, 0x20);
    auto u32 = BufferReader{Slice{data}}.Int<uint32_t>(Endianness::kBig);
    EXPECT_FALSE(u32);
    auto i32 = BufferReader{Slice{data}}.Int<int32_t>(Endianness::kBig);
    EXPECT_FALSE(i32);
}

TEST(BufferReaderTest, IntegerLittleEndian) {
    auto data = Data(0x10, 0x20, 0x30, 0x40);
    auto u32 = BufferReader{Slice{data}}.Int<uint32_t>(Endianness::kLittle);
    EXPECT_EQ(u32, 0x40302010u);
    auto i32 = BufferReader{Slice{data}}.Int<int32_t>(Endianness::kLittle);
    EXPECT_EQ(i32, 0x40302010);
}

TEST(BufferReaderTest, IntegerLittleEndian_TooShort) {
    auto data = Data(0x30, 0x40);
    auto u32 = BufferReader{Slice{data}}.Int<uint32_t>(Endianness::kLittle);
    EXPECT_FALSE(u32);
    auto i32 = BufferReader{Slice{data}}.Int<int32_t>(Endianness::kLittle);
    EXPECT_FALSE(i32);
}

TEST(BufferReaderTest, Float) {
    auto data = Data(0x00, 0x00, 0x08, 0x41);
    auto f32 = BufferReader{Slice{data}}.Float<float>();
    ASSERT_TRUE(f32);
    EXPECT_EQ(f32.Get(), 8.5f);
}

TEST(BufferReaderTest, Float_TooShort) {
    auto data = Data(0x08, 0x41);
    auto f32 = BufferReader{Slice{data}}.Float<float>();
    EXPECT_FALSE(f32);
}

}  // namespace
}  // namespace tint::bytes
