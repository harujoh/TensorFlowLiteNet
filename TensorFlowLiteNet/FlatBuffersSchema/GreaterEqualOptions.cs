// <auto-generated>
//  automatically generated by the FlatBuffers compiler, do not modify
// </auto-generated>

namespace tflite
{

using global::System;
using global::System.Collections.Generic;
using global::FlatBuffers;

public struct GreaterEqualOptions : IFlatbufferObject
{
  private Table __p;
  public ByteBuffer ByteBuffer { get { return __p.bb; } }
  public static void ValidateVersion() { FlatBufferConstants.FLATBUFFERS_1_12_0(); }
  public static GreaterEqualOptions GetRootAsGreaterEqualOptions(ByteBuffer _bb) { return GetRootAsGreaterEqualOptions(_bb, new GreaterEqualOptions()); }
  public static GreaterEqualOptions GetRootAsGreaterEqualOptions(ByteBuffer _bb, GreaterEqualOptions obj) { return (obj.__assign(_bb.GetInt(_bb.Position) + _bb.Position, _bb)); }
  public void __init(int _i, ByteBuffer _bb) { __p = new Table(_i, _bb); }
  public GreaterEqualOptions __assign(int _i, ByteBuffer _bb) { __init(_i, _bb); return this; }


  public static void StartGreaterEqualOptions(FlatBufferBuilder builder) { builder.StartTable(0); }
  public static Offset<tflite.GreaterEqualOptions> EndGreaterEqualOptions(FlatBufferBuilder builder) {
    int o = builder.EndTable();
    return new Offset<tflite.GreaterEqualOptions>(o);
  }
};


}