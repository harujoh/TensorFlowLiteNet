// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct Tensor : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static Tensor GetRootAsTensor(ByteBuffer _bb) { return GetRootAsTensor(_bb, new Tensor()); }
  public static Tensor GetRootAsTensor(ByteBuffer _bb, Tensor obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public Tensor __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int Shape(int j) { int o = __p.__offset(4); return o != 0 ? __p.bb.GetInt(__p.__vector(o) + j * 4) : (int)0; }
  public int ShapeLength { get { int o = __p.__offset(4); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<int> GetShapeBytes() { return __p.__vector_as_span<int>(4, 4); }
#else
  public ArraySegment<byte>? GetShapeBytes() { return __p.__vector_as_arraysegment(4); }
#endif
  public int[] GetShapeArray() { return __p.__vector_as_array<int>(4); }
  public tflite.TensorType Type { get { int o = __p.__offset(6); return o != 0 ? (tflite.TensorType)__p.bb.GetSbyte(o + __p.bb_pos) : tflite.TensorType.FLOAT32; } }
  public uint Buffer { get { int o = __p.__offset(8); return o != 0 ? __p.bb.GetUint(o + __p.bb_pos) : (uint)0; } }
  public string Name { get { int o = __p.__offset(10); return o != 0 ? __p.__string(o + __p.bb_pos) : null; } }
#if ENABLE_SPAN_T
  public Span<byte> GetNameBytes() { return __p.__vector_as_span<byte>(10, 1); }
#else
  public ArraySegment<byte>? GetNameBytes() { return __p.__vector_as_arraysegment(10); }
#endif
  public byte[] GetNameArray() { return __p.__vector_as_array<byte>(10); }
  public tflite.QuantizationParameters? Quantization { get { int o = __p.__offset(12); return o != 0 ? (tflite.QuantizationParameters?)(new tflite.QuantizationParameters()).__assign(__p.__indirect(o + __p.bb_pos), __p.bb) : null; } }
  public bool IsVariable { get { int o = __p.__offset(14); return o != 0 ? 0!=__p.bb.Get(o + __p.bb_pos) : (bool)false; } }
  public tflite.SparsityParameters? Sparsity { get { int o = __p.__offset(16); return o != 0 ? (tflite.SparsityParameters?)(new tflite.SparsityParameters()).__assign(__p.__indirect(o + __p.bb_pos), __p.bb) : null; } }
  public int ShapeSignature(int j) { int o = __p.__offset(18); return o != 0 ? __p.bb.GetInt(__p.__vector(o) + j * 4) : (int)0; }
  public int ShapeSignatureLength { get { int o = __p.__offset(18); return o != 0 ? __p.__vector_len(o) : 0; } }
#if ENABLE_SPAN_T
  public Span<int> GetShapeSignatureBytes() { return __p.__vector_as_span<int>(18, 4); }
#else
  public ArraySegment<byte>? GetShapeSignatureBytes() { return __p.__vector_as_arraysegment(18); }
#endif
  public int[] GetShapeSignatureArray() { return __p.__vector_as_array<int>(18); }

  public static Offset<tflite.Tensor> CreateTensor(FlatBufferBuilder builder,
      VectorOffset shapeOffset = default(VectorOffset),
      tflite.TensorType type = tflite.TensorType.FLOAT32,
      uint buffer = 0,
      StringOffset nameOffset = default(StringOffset),
      Offset<tflite.QuantizationParameters> quantizationOffset = default(Offset<tflite.QuantizationParameters>),
      bool is_variable = false,
      Offset<tflite.SparsityParameters> sparsityOffset = default(Offset<tflite.SparsityParameters>),
      VectorOffset shape_signatureOffset = default(VectorOffset)) {
    builder.StartTable(8);
    Tensor.AddShapeSignature(builder, shape_signatureOffset);
    Tensor.AddSparsity(builder, sparsityOffset);
    Tensor.AddQuantization(builder, quantizationOffset);
    Tensor.AddName(builder, nameOffset);
    Tensor.AddBuffer(builder, buffer);
    Tensor.AddShape(builder, shapeOffset);
    Tensor.AddIsVariable(builder, is_variable);
    Tensor.AddType(builder, type);
    return Tensor.EndTensor(builder);
  }

  public static void StartTensor(FlatBufferBuilder builder) { builder.StartTable(8); }
  public static void AddShape(FlatBufferBuilder builder, VectorOffset shapeOffset) { builder.AddOffset(0, shapeOffset.Value, 0); }
  public static VectorOffset CreateShapeVector(FlatBufferBuilder builder, int[] data) { builder.StartVector(4, data.Length, 4); for (int i = data.Length - 1; i >= 0; i--) builder.AddInt(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateShapeVectorBlock(FlatBufferBuilder builder, int[] data) { builder.StartVector(4, data.Length, 4); builder.Add(data); return builder.EndVector(); }
  public static void StartShapeVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(4, numElems, 4); }
  public static void AddType(FlatBufferBuilder builder, tflite.TensorType type) { builder.AddSbyte(1, (sbyte)type, 0); }
  public static void AddBuffer(FlatBufferBuilder builder, uint buffer) { builder.AddUint(2, buffer, 0); }
  public static void AddName(FlatBufferBuilder builder, StringOffset nameOffset) { builder.AddOffset(3, nameOffset.Value, 0); }
  public static void AddQuantization(FlatBufferBuilder builder, Offset<tflite.QuantizationParameters> quantizationOffset) { builder.AddOffset(4, quantizationOffset.Value, 0); }
  public static void AddIsVariable(FlatBufferBuilder builder, bool isVariable) { builder.AddBool(5, isVariable, false); }
  public static void AddSparsity(FlatBufferBuilder builder, Offset<tflite.SparsityParameters> sparsityOffset) { builder.AddOffset(6, sparsityOffset.Value, 0); }
  public static void AddShapeSignature(FlatBufferBuilder builder, VectorOffset shapeSignatureOffset) { builder.AddOffset(7, shapeSignatureOffset.Value, 0); }
  public static VectorOffset CreateShapeSignatureVector(FlatBufferBuilder builder, int[] data) { builder.StartVector(4, data.Length, 4); for (int i = data.Length - 1; i >= 0; i--) builder.AddInt(data[i]); return builder.EndVector(); }
  public static VectorOffset CreateShapeSignatureVectorBlock(FlatBufferBuilder builder, int[] data) { builder.StartVector(4, data.Length, 4); builder.Add(data); return builder.EndVector(); }
  public static void StartShapeSignatureVector(FlatBufferBuilder builder, int numElems) { builder.StartVector(4, numElems, 4); }
  public static Offset<tflite.Tensor> EndTensor(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.Tensor>(o);
  }
};


}
