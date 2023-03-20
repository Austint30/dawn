diagnostic_filtering/loop_attribute.wgsl:5:9 warning: 'dpdx' must only be called from uniform control flow
    _ = dpdx(1.0);
        ^^^^^^^^^

diagnostic_filtering/loop_attribute.wgsl:7:7 note: control flow depends on possibly non-uniform value
      break if x > 0.0;
      ^^^^^

diagnostic_filtering/loop_attribute.wgsl:7:16 note: user-defined input 'x' of 'main' may be non-uniform
      break if x > 0.0;
               ^

struct tint_symbol_1 {
  float x : TEXCOORD0;
};

void main_inner(float x) {
  while (true) {
    {
      if ((x > 0.0f)) { break; }
    }
  }
}

void main(tint_symbol_1 tint_symbol) {
  main_inner(tint_symbol.x);
  return;
}