// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct QuantizationParameters : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static QuantizationParameters GetRootAsQuantizationParameters(ByteBuffer _bb) { return GetRootAsQuantizationParameters(_bb, new QuantizationParameters()); }
  public static QuantizationParameters GetRootAsQuantizationParameters(ByteBuffer _bb, QuantizationParameters obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public QuantizationParameters __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public float Min(int j) { int o = __p.__offset(4); return o != 0 ? __p.bb.GetFloat(__p.__vector(o) + j * 4) : (float)0; }
  public int MinLength { get { int o = __p.__offset(4); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<float> GetMinBytes() { return __p.__vector_as_span<float>(4, 4); }
#else
  public ArraySegment<byte>? GetMinBytes() { return __p.__vector_as_arraysegment(4); }
#endif
  public float[] GetMinArray() { return __p.__vector_as_array<float>(4); }
  public float Max(int j) { int o = __p.__offset(6); return o != 0 ? __p.bb.GetFloat(__p.__vector(o) + j * 4) : (float)0; }
  public int MaxLength { get { int o = __p.__offset(6); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<float> GetMaxBytes() { return __p.__vector_as_span<float>(6, 4); }
#else
  public ArraySegment<byte>? GetMaxBytes() { return __p.__vector_as_arraysegment(6); }
#endif
  public float[] GetMaxArray() { return __p.__vector_as_array<float>(6); }
  public float Scale(int j) { int o = __p.__offset(8); return o != 0 ? __p.bb.GetFloat(__p.__vector(o) + j * 4) : (float)0; }
  public int ScaleLength { get { int o = __p.__offset(8); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<float> GetScaleBytes() { return __p.__vector_as_span<float>(8, 4); }
#else
  public ArraySegment<byte>? GetScaleBytes() { return __p.__vector_as_arraysegment(8); }
#endif
  public float[] GetScaleArray() { return __p.__vector_as_array<float>(8); }
  public long ZeroPoint(int j) { int o = __p.__offset(10); return o != 0 ? __p.bb.GetLong(__p.__vector(o) + j * 8) : (long)0; }
  public int ZeroPointLength { get { int o = __p.__offset(10); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<long> GetZeroPointBytes() { return __p.__vector_as_span<long>(10, 8); }
#else
  public ArraySegment<byte>? GetZeroPointBytes() { return __p.__vector_as_arraysegment(10); }
#endif
  public long[] GetZeroPointArray() { return __p.__vector_as_array<long>(10); }
  public tflite.QuantizationDetails DetailsType { get { int o = __p.__offset(12); return o != 0 ? (tflite.QuantizationDetails)__p.bb.Get(o + __p.bb_pos) : tflite.QuantizationDetails.NONE; } }
  public TTable? Details<TTable>() where TTable : struct, IFlatbufferObject { int o = __p.__offset(14); return o != 0 ? (TTable?)__p.__union<TTable>(o + __p.bb_pos) : null; }
  public int QuantizedDimension { get { int o = __p.__offset(16); return o != 0 ? __p.bb.GetInt(o + __p.bb_pos) : (int)0; } }

  public static Offset<tflite.QuantizationParameters> CreateQuantizationParameters(FlatBufferBuilder builder,
      VectorOffset minOffset = default(VectorOffset),
      VectorOffset maxOffset = default(VectorOffset),
      VectorOffset scaleOffset = default(VectorOffset),
      VectorOffset zero_pointOffset = default(VectorOffset),
      tflite.QuantizationDetails details_type = tflite.QuantizationDetails.NONE,
      int detailsOffset = 0,
      int quantized_dimension = 0) {
    builder.StartTable(7);
    QuantizationParameters.AddQuantizedDimension(builder, quantized_dimension);
    QuantizationParameters.AddDetails(builder, detailsOffset);
    QuantizationParameters.AddZeroPoint(builder, zero_pointOffset);
    QuantizationParameters.AddScale(builder, scaleOffset);
    QuantizationParameters.AddMax(builder, maxOffset);
    QuantizationParameters.AddMin(builder, minOffset);
    QuantizationParameters.AddDetailsType(builder, details_type);
    return QuantizationParameters.EndQuantizationParameters(builder);
  }

  public static void StartQuantizationParameters(FlatBufferBuilder builder) { builder.StartTable(7); }
  public static void AddMin(FlatBufferBuilder builder, VectorOffset minOffset) { builder.AddOffset(0, minOffset.Value, 0); }
  public static VectorOffset CreateMinVector(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); for (int i = data.Length - 1; i >= 0; i--) builder.AddFloat(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateMinVectorBlock(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); builder.Add(data); return builder.EndVector(); }
  public static void StartMinVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(4, numElems, 4); }
  public static void AddMax(FlatBufferBuilder builder, VectorOffset maxOffset) { builder.AddOffset(1, maxOffset.Value, 0); }
  public static VectorOffset CreateMaxVector(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); for (int i = data.Length - 1; i >= 0; i--) builder.AddFloat(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateMaxVectorBlock(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); builder.Add(data); return builder.EndVector(); }
  public static void StartMaxVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(4, numElems, 4); }
  public static void AddScale(FlatBufferBuilder builder, VectorOffset scaleOffset) { builder.AddOffset(2, scaleOffset.Value, 0); }
  public static VectorOffset CreateScaleVector(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); for (int i = data.Length - 1; i >= 0; i--) builder.AddFloat(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateScaleVectorBlock(FlatBufferBuilder builder, float[] data) { builder.StartVector(4, data.Length, 4); builder.Add(data); return builder.EndVector(); }
  public static void StartScaleVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(4, numElems, 4); }
  public static void AddZeroPoint(FlatBufferBuilder builder, VectorOffset zeroPointOffset) { builder.AddOffset(3, zeroPointOffset.Value, 0); }
  public static VectorOffset CreateZeroPointVector(FlatBufferBuilder builder, long[] data) { builder.StartVector(8, data.Length, 8); for (int i = data.Length - 1; i >= 0; i--) builder.AddLong(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateZeroPointVectorBlock(FlatBufferBuilder builder, long[] data) { builder.StartVector(8, data.Length, 8); builder.Add(data); return builder.EndVector(); }
  public static void StartZeroPointVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(8, numElems, 8); }
  public static void AddDetailsType(FlatBufferBuilder builder, tflite.QuantizationDetails detailsType) { builder.AddByte(4, (byte)detailsType, 0); }
  public static void AddDetails(FlatBufferBuilder builder, int detailsOffset) { builder.AddOffset(5, detailsOffset, 0); }
  public static void AddQuantizedDimension(FlatBufferBuilder builder, int quantizedDimension) { builder.AddInt(6, quantizedDimension, 0); }
  public static Offset<tflite.QuantizationParameters> EndQuantizationParameters(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.QuantizationParameters>(o);
  }
};


}