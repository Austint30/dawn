Texture2DArray<float4> t_f : register(t0);
Texture2DArray<int4> t_i : register(t1);
Texture2DArray<uint4> t_u : register(t2);

[numthreads(1, 1, 1)]
void main() {
  uint4 tint_tmp;
  t_f.GetDimensions(1, tint_tmp.x, tint_tmp.y, tint_tmp.z, tint_tmp.w);
  uint2 fdims = tint_tmp.xy;
  uint4 tint_tmp_1;
  t_i.GetDimensions(1, tint_tmp_1.x, tint_tmp_1.y, tint_tmp_1.z, tint_tmp_1.w);
  uint2 idims = tint_tmp_1.xy;
  uint4 tint_tmp_2;
  t_u.GetDimensions(1, tint_tmp_2.x, tint_tmp_2.y, tint_tmp_2.z, tint_tmp_2.w);
  uint2 udims = tint_tmp_2.xy;
  return;
}
