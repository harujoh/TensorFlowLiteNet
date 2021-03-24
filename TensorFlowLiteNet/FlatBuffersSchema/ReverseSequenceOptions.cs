// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct ReverseSequenceOptions : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static ReverseSequenceOptions GetRootAsReverseSequenceOptions(ByteBuffer _bb) { return GetRootAsReverseSequenceOptions(_bb, new ReverseSequenceOptions()); }
  public static ReverseSequenceOptions GetRootAsReverseSequenceOptions(ByteBuffer _bb, ReverseSequenceOptions obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public ReverseSequenceOptions __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }

  public int SeqDim { get { int o = __p.__offset(4); return o != 0 ? __p.bb.GetInt(o + __p.bb_pos) : (int)0; } }
  public int BatchDim { get { int o = __p.__offset(6); return o != 0 ? __p.bb.GetInt(o + __p.bb_pos) : (int)0; } }

  public static Offset<tflite.ReverseSequenceOptions> CreateReverseSequenceOptions(FlatBufferBuilder builder,
      int seq_dim = 0,
      int batch_dim = 0) {
    builder.StartTable(2);
    ReverseSequenceOptions.AddBatchDim(builder, batch_dim);
    ReverseSequenceOptions.AddSeqDim(builder, seq_dim);
    return ReverseSequenceOptions.EndReverseSequenceOptions(builder);
  }

  public static void StartReverseSequenceOptions(FlatBufferBuilder builder) { builder.StartTable(2); }
  public static void AddSeqDim(FlatBufferBuilder builder, int seqDim) { builder.AddInt(0, seqDim, 0); }
  public static void AddBatchDim(FlatBufferBuilder builder, int batchDim) { builder.AddInt(1, batchDim, 0); }
  public static Offset<tflite.ReverseSequenceOptions> EndReverseSequenceOptions(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.ReverseSequenceOptions>(o);
  }
};


}
