// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct LogicalOrOptions : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static LogicalOrOptions GetRootAsLogicalOrOptions(ByteBuffer _bb) { return GetRootAsLogicalOrOptions(_bb, new LogicalOrOptions()); }
  public static LogicalOrOptions GetRootAsLogicalOrOptions(ByteBuffer _bb, LogicalOrOptions obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public LogicalOrOptions __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }


  public static void StartLogicalOrOptions(FlatBufferBuilder builder) { builder.StartTable(0); }
  public static Offset<tflite.LogicalOrOptions> EndLogicalOrOptions(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.LogicalOrOptions>(o);
  }
};


}
