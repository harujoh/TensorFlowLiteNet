// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct SliceOptions : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static SliceOptions GetRootAsSliceOptions(ByteBuffer _bb) { return GetRootAsSliceOptions(_bb, new SliceOptions()); }
  public static SliceOptions GetRootAsSliceOptions(ByteBuffer _bb, SliceOptions obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public SliceOptions __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }


  public static void StartSliceOptions(FlatBufferBuilder builder) { builder.StartTable(0); }
  public static Offset<tflite.SliceOptions> EndSliceOptions(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.SliceOptions>(o);
  }
};


}
