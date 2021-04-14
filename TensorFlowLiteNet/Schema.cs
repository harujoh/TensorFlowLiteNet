using System.Collections.Generic;
using System.Text;
using FlatBuffers;
using tflite;

namespace Schema
{
    // 新しい量子化技術を試験するためのカスタム量子化パラメータ
    class CustomQuantization
    {
        public byte[] Custom;
    }

    // 特定の量子化手法のパラメータを表します
    enum QuantizationDetails //union
    {
        CustomQuantization,
    }

    // 量子化されたテンソルをfloatに戻すためのパラメータです
    class QuantizationParameters
    {
        // この4つのパラメータが、非対称線形量子化パラメータです
        // 量子化された値qに対応するフロート値fは以下のようにしてください
        //   f = scale * (q - zero_point)
        // 他の量子化タイプの場合は、以下のUantizationDetailsを使用してください
        List<float> Min = new List<float>();  // tensorflowへのインポート用
        List<float> Max = new List<float>();  // tensorflowへのインポート用
        List<float> Scale = new List<float>();  // テンソル値の量子化用
        List<long> ZeroPoint = new List<long>();

        // これがNullでない場合、他の量子化パラメータ（min、max、scale、zero_pointフィールドの上）は無視され、
        // QuantizationDetailsのunionの値が使用されます
        QuantizationDetails Details;

        // scalesとzero_pointsが対応するテンソル形状の次元を指定します。
        // たとえば、dims=[4, 3, 2, 1]のテンソルtに量子化パラメータを指定する場合は
        //   scale=[1.0, 2.0, 3.0], zero_point=[1, 2, 3], quantization_dimension=1
        // そしてtの2次元目に沿って量子化されます。
        //   t[:, 0, :, :] は scale[0]=1.0, zero_point[0]=1 を持ち
        //   t[:, 1, :, :] は scale[1]=2.0, zero_point[0]=2 を持ち
        //   t[:, 2, :, :] は scale[2]=3.0, zero_point[0]=3 を持つ
        int QuantizedDimension;
    }

    // 疎なテンソル
    // TACOフォーマットの改良版を使用しています。
    // 参考: http://tensor-compiler.org/kjolstad-oopsla17-tensor-compiler.pdf
    //
    // 例えばdims (d0, ..., dn-1)を持つn次元の密なテンソルを、潜在的にdims (dn, ..., dn+k-1)を持つ
    // k次元のブロック(0 <= k <= n)と一緒にエンコードするには、フォーマットで指定する必要があります。
    //   1. どのような順序でこれらの次元を横断するか。
    //      例えば、2次元行列を行優先で格納する場合、走査順序は（d0, d1）となり、
    //      列優先で格納する場合、走査順序は（d1, d0）となります。
    //      また、2次元行列に2次元の内部ブロックがある場合、走査順序は（d0, d1, d2, d3）となります。
    //   2. (dn, _, dn+k-1)の各ブロック次元が、(d0, _, dn-1)の元のテンソル次元に
    //      どのようにマッピングされるか
    //   3. 上記で定義された走査順序で、各次元のフォーマット(dense vs. sparse)と
    //      インデックスのメタデータを指定します。
    //      密な次元の場合、これは単にその次元のサイズです。
    //      疎な次元の場合は、CSR（Compressed Sparse Row）形式で定義された圧縮インデックスと同じです。
    //      (http://scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html)

    // ディメンションのストレージタイプ
    // 現在は以下をサポートしています
    //  1. DENSE: このディメンジョンの各座標は暗黙的に保存されます
    //  2. SPARSE_CSR：非ゼロの要素を持つ座標のみが格納されます
    // 圧縮技術は、CSRが使用するものと同じです。
    // 将来的には、異なる圧縮技術を持つスパース・ディメンションのようなタイプが
    // 追加される可能性があります。
    enum DimensionType : byte
    {
        DENSE = 0,
        SPARSE_CSR = 1,
    }

    class Int32Vector
    {
        public List<int> Values = new List<int>();
    }

    class Uint16Vector
    {
        public List<ushort> Values = new List<ushort>();
    }

    class Uint8Vector
    {
        public List<byte> Values = new List<byte>();
    }

    // 疎な次元のインデックスメタデータを格納するための可変バッファ
    // テンソルの形状がint32のベクトルであるため、最も広い型はUInt32ではなくInt32である
    // 次元ごとのインデックスがこの範囲をオーバーフローしないようにします
    enum SparseIndexVector //union
    {
        Int32Vector,
        Uint16Vector,
        Uint8Vector
    }

    class DimensionMetadata
    {
        // 次元が疎か密か
        public DimensionType format;
        // 次元に適用されるインデックスメタデータ。
        //   - formatがDimensionType.DENSEの場合は、dense_sizeフィールドを使用して 
        //     そのディメンジョンのサイズを格納します。
        //     そのディメンジョンの各インデックスは 暗黙的に格納されます。
        //   - formatがDimensionType.SPARSE_CSRの場合は、array_segmentsとarray_indicesを使用して、
        //     その次元をエンコードします。
        //     array_segmentsは、インデックス配列をどのようにセグメント化するかを表し、
        //     各セグメントは前の次元の1つの要素に対応します。
        //     array_indicesは、この次元内のゼロではない要素のインデックスを表します
        //     (CSR行列フォーマットのように、最初の配列が行ポインタで、2番目の配列が列インデックスです)
        public int DenseSize;
        public SparseIndexVector ArraySegments;
        public SparseIndexVector ArrayIndices;
    }

    // 疎なTfLiteテンソルをエンコードするためのパラメータ
    class SparsityParameters
    {
        // 以下は密なテンソルの`shape`フィールドで定義された次元のトラバーサル順序です。
        // 次元(d0, d1, ..., dn-1)を持つn次元テンソルの場合
        //   - ブロックスパースでない場合TraversalOrderは(d0, ..., dn-1)の順列にすぎません。
        //     例えば，行優先で格納された2次元行列はTraversalOrder = (d0, d1) となります
        //   - k次元のブロック(0 <= k <= n)でブロックスパースの場合TraversalOrderはn + k個の要素を持ちます
        //     最初のn個の要素は明らかに(d0, ..., dn-1)の順列です。
        //     ラスクk個の要素は(dn, ..., dn+k-1)の順列で、ブロックの内部でのトラバース方法を定義しています
        //     例えば、2次元ブロックを持つ2次元行列で、両方とも行優先に格納されている場合TraversalOrder = (d0, d1, d2, d3)となります。  
        public List<int> TraversalOrder = new List<int>();

        // k次元のブロックを持つn次元テンソル(0 <= k <= n)に対して(dn, ..., dn+k-1)のブロック次元が
        // (d0, ..., dn)の元のテンソル次元にどのように対応するかを格納します
        // それは(dn, ..., dn+k-1)の順に格納されます。
        // block-sparseでない場合，このフィールドはNULLです。
        public List<int> BlockMap = new List<int>();

        // 上で定義された走査順序で、元の密なテンソルの中のゼロでない値を探すために、各次元に必要なメタデータを指定します。
        // dim_metadata配列のサイズ＝traversal_order配列のサイズ＝n＋k
        List<DimensionMetadata> DimMetadata = new List<DimensionMetadata>();
    }

    class Tensor
    {
        // テンソルの形状
        // 各エントリの意味は演算子に依存しますが、組み込みの演算子は次の使用を想定します。
        // [バッチサイズ、高さ、幅、チャンネル数](これはTensorflowのNHWCに相当)
        public int[] Shape;
        public TensorType Type;

        // モデルのルートにあるバッファーテーブルを参照するインデックスです。
        // また、関連するデータバッファ（つまり中間結果）がない場合、
        // これは0（常に存在する空のバッファを意味する）です。
        //
        // data_buffer自体は非透過コンテナで、ターゲットデバイスがリトルエンディアンであることを前提としています。
        // さらに、すべての組み込み演算子は、メモリが次のように順序付けられていることを想定しています:
        // shapeが [4, 3, 2] の場合 index[i, j, k] は data_buffer[i*3*2 + j*2 + k] にマッピングされます。
        public uint Buffer;
        public string Name; //デバッグやtensorflowへのインポートに使用します。
        QuantizationParameters Quantization; //任意

        private bool IsVariable = false;

        // 疎なテンソルをエンコードするためのパラメータです。
        // tensorflow/lite/testdata/sparse_tensor.jsonの例を参照してください。
        SparsityParameters Sparsity; //任意

        // 未知の次元を持つ `shape` をエンコードします。
        // 未知の次元は -1 で表されます。
        int[] ShapeSignature; // 任意

        public Tensor(
            int[] shape = null,
            TensorType type = TensorType.FLOAT32,
            uint buffer = 0,
            string name = "",
            QuantizationParameters quantization = null,
            bool is_variable = false,
            SparsityParameters sparsity = null,
            int[] shape_signature = null)
        {
            this.Shape = shape;
            this.Type = type;
            this.Buffer = buffer;
            this.Name = name;
            this.Quantization = quantization;
            this.IsVariable = is_variable;
            this.Sparsity = sparsity;
            this.ShapeSignature = shape_signature;
        }

        public Offset<tflite.Tensor> GetOffset(FlatBufferBuilder fbb)
        {
            VectorOffset shapeOffsetVector = tflite.Tensor.CreateShapeVector(fbb, this.Shape);
            StringOffset nameOffset = fbb.CreateString(this.Name);
            Offset<tflite.QuantizationParameters> quantizationOffsetVector = tflite.QuantizationParameters.CreateQuantizationParameters(fbb, default, default, default, default, tflite.QuantizationDetails.NONE, 0, 0);
            return tflite.Tensor.CreateTensor(fbb, shapeOffsetVector, this.Type, this.Buffer, nameOffset, quantizationOffsetVector, this.IsVariable);
        }
    }

    interface IBuiltinOptions
    {
        BuiltinOptions Type { get; }
        int GetValue(FlatBufferBuilder fbb);
    }

    class AddOptions: IBuiltinOptions
    {
        public BuiltinOptions Type => BuiltinOptions.AddOptions;
        public ActivationFunctionType FusedActivationFunction;
        public bool PotScaleInt16;

        public AddOptions(ActivationFunctionType fused_activation_function = ActivationFunctionType.NONE, bool pot_scale_int16 = true)
        {
            this.FusedActivationFunction = fused_activation_function;
            this.PotScaleInt16 = pot_scale_int16;
        }

        public int GetValue(FlatBufferBuilder fbb)
        {
            return tflite.AddOptions.CreateAddOptions(fbb, this.FusedActivationFunction, this.PotScaleInt16).Value;
        }
    }

    class SubOptions : IBuiltinOptions
    {
        public BuiltinOptions Type => BuiltinOptions.SubOptions;
        public ActivationFunctionType FusedActivationFunction;
        public bool PotScaleInt16;

        public SubOptions(ActivationFunctionType fused_activation_function = ActivationFunctionType.NONE, bool pot_scale_int16 = true)
        {
            this.FusedActivationFunction = fused_activation_function;
            this.PotScaleInt16 = pot_scale_int16;
        }

        public int GetValue(FlatBufferBuilder fbb)
        {
            return tflite.SubOptions.CreateSubOptions(fbb, this.FusedActivationFunction, this.PotScaleInt16).Value;
        }
    }

    class MulOptions : IBuiltinOptions
    {
        public BuiltinOptions Type => BuiltinOptions.MulOptions;
        public ActivationFunctionType FusedActivationFunction;

        public MulOptions(ActivationFunctionType fused_activation_function = ActivationFunctionType.NONE)
        {
            this.FusedActivationFunction = fused_activation_function;
        }

        public int GetValue(FlatBufferBuilder fbb)
        {
            return tflite.MulOptions.CreateMulOptions(fbb, this.FusedActivationFunction).Value;
        }
    }

    class DivOptions : IBuiltinOptions
    {
        public BuiltinOptions Type => BuiltinOptions.DivOptions;
        public ActivationFunctionType FusedActivationFunction;

        public DivOptions(ActivationFunctionType fused_activation_function = ActivationFunctionType.NONE)
        {
            this.FusedActivationFunction = fused_activation_function;
        }

        public int GetValue(FlatBufferBuilder fbb)
        {
            return tflite.DivOptions.CreateDivOptions(fbb, this.FusedActivationFunction).Value;
        }
    }

    // OperatorCodeは、オペレータがビルトインの場合は列挙された値（BuiltinOperator）
    // オペレータがカスタムの場合は文字列となります。
    class OperatorCode
    {
        // このフィールドは後方互換性のためのものです。
        // このフィールドは、builtin_codeフィールドの値が、
        // BulitinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES[=127]よりも小さい場合に使用される。
        //
        // 名前はDeprecatedだが普通に使用されている。
        // BuiltinCodeがADD(=0)に設定され、こちらの値が使用されるパターンが多い
        public sbyte DeprecatedBuiltinCode;
        public string CustomCode;

        // 演算子のバージョンです。
        // オペレータに新しいパラメータが導入されるたびに、バージョンを更新する必要があります。
        public int Version;

        // このフィールドは、ビルトインオペレータの表現枯渇問題を解決するために導入されました
        // (元のBuiltinOperator enumフィールドはバイトで表現されていた為これが枯渇した)
        // このフィールドは、builtin_codeフィールドの値が
        // BulitinOperator.PLACEHOLDER_FOR_GREATER_OP_CODES[=127]よりも大きい場合に使用される。
        public BuiltinOperator BuiltinCode;

        public OperatorCode(
            sbyte deprecated_builtin_code = 0,
            string custom_code = "",
            int version = 1,
            BuiltinOperator builtin_code = BuiltinOperator.ADD)
        {
            this.DeprecatedBuiltinCode = deprecated_builtin_code;
            this.CustomCode = custom_code;
            this.Version = version;
            this.BuiltinCode = builtin_code;
        }

        public Offset<tflite.OperatorCode> GetOffset(FlatBufferBuilder fbb)
        {
            StringOffset custom_codeOffset = CustomCode != "" ? fbb.CreateString(CustomCode) : default;
            return tflite.OperatorCode.CreateOperatorCode(fbb, this.DeprecatedBuiltinCode, custom_codeOffset, this.Version, this.BuiltinCode);
        }
    }

    // 演算子はテンソルを入力と出力として受け取ります。
    // 実行される演算の種類は、有効なOperatorCodesのリストへのインデックスによって決定され、
    // 各演算の詳細はBuiltinOptionsまたはCustomOptionsを使用して設定されます。
    class Operator
    {
        // operator_codes配列へのインデックス
        // ここで整数を使用すると、マップの検索が複雑になるのを避けることができます。
        public uint OpecodeIndex;

        // オプション入力は-1で表示されます
        public int[] Inputs;
        public int[] Outputs;

        public sbyte[] CustomOptions;
        public CustomOptionsFormat CustomOptionsFormat;

        // この演算子によって内容が変更される入力テンソルを示すフラグのリストです。
        // (例えば、RNNやLSTMで使用されます)
        // 例えばinputs配列が5つのテンソルを参照しており、2番目と5番目が変更可能な変数である場合
        // このリストは[false, true, false, false, true]になります
        //
        // リストが空の場合、この演算子ではどの変数も変更しません。
        // リストはinputsと同じ長さになるか、空になります。
        private bool[] MutatingVariableInputs;

        // Opの内部にあるサブグラフのテンソルへのインデックスのリストです。
        // 内部テンソルとは、演算で入出力されない、内部の計算の一部であるテンソルのことです。
        // そのため、オペレーションの実装では、メモリをより効率的に管理することができます。
        // しかし、量子化パラメータなどの関連するメタデータを必要とする計算の一部であるため、
        // これらは必要とされます(つまり、単なる実装の詳細ではありません)
        private int[] Intermediates;

        //本家にはない項目
        private IBuiltinOptions BuiltinOption;

        public Operator(
            uint opcode_index = 0,
            int[] inputs = null,
            int[] outputs = null,
            IBuiltinOptions builtin_options = null,
            sbyte[] custom_options = null,
            CustomOptionsFormat custom_options_format = CustomOptionsFormat.FLEXBUFFERS,
            bool[] mutating_variable_inputs = null,
            int[] intermediates = null)
        {
            this.OpecodeIndex = opcode_index;
            this.Inputs = inputs;
            this.Outputs = outputs;
            this.BuiltinOption = builtin_options;
            this.CustomOptions = custom_options;
            this.CustomOptionsFormat = custom_options_format;
            this.MutatingVariableInputs = mutating_variable_inputs;
            this.Intermediates = intermediates;
        }

        public Offset<tflite.Operator> GetOffset(FlatBufferBuilder fbb)
        {
            //subgraphs-operators-inputs
            VectorOffset operatorInputsOffsetVector = SubGraph.CreateInputsVector(fbb, Inputs);

            //subgraphs-operators-inputs
            VectorOffset operatorOutputsOffsetVector = SubGraph.CreateOutputsVector(fbb, Outputs);

            //事前に用意しないとエラーが出る
            int builtinOptionValue = BuiltinOption.GetValue(fbb);

            tflite.Operator.StartOperator(fbb);
            tflite.Operator.AddOpcodeIndex(fbb, this.OpecodeIndex);
            tflite.Operator.AddInputs(fbb, operatorInputsOffsetVector);
            tflite.Operator.AddOutputs(fbb, operatorOutputsOffsetVector);
            tflite.Operator.AddBuiltinOptionsType(fbb, this.BuiltinOption.Type);
            tflite.Operator.AddBuiltinOptions(fbb, builtinOptionValue);
            tflite.Operator.AddCustomOptionsFormat(fbb, this.CustomOptionsFormat);
            return tflite.Operator.EndOperator(fbb);
        }

    }

    // サブグラフを定義するルートタイプで、通常はモデル全体を表します。
    class Subgraph
    {
        // このサブグラフで使用されているすべてのテンソルのリストです。
        public Tensor[] Tensors;

        // このサブグラフへの入力であるテンソルのインデックス。
        // これは、推論のためにサブグラフに入力される非静的なテンソルのリストであることに注意してください。
        public int[] Inputs;

        // このサブグラフから出力されたテンソルのインデックス。
        // これは、サブグラフの推論の積とみなされる出力テンソルのリストであることに注意してください。
        public int[] Outputs;

        // すべてのオペレーターが実行順で格納される
        public Operator[] Operators;

        // このサブグラフの名前（デバッグ時に使用）
        public string Name;

        public Subgraph(
            Tensor[] tensors = null,
            int[] inputs = null,
            int[] outputs = null,
            Operator[] operators = null,
            string name = "")
        {
            this.Tensors = tensors;
            this.Inputs = inputs;
            this.Outputs = outputs;
            this.Operators = operators;
            this.Name = name;
        }

        public Offset<SubGraph> GetOffset(FlatBufferBuilder fbb)
        {
            Offset<tflite.Operator>[] graphOperatorsOffset = new Offset<tflite.Operator>[Operators.Length];
            for (int i = 0; i < graphOperatorsOffset.Length; i++)
            {
                graphOperatorsOffset[i] = Operators[i].GetOffset(fbb);
            }

            VectorOffset graphOperatorsOffsetVector = SubGraph.CreateOperatorsVector(fbb, graphOperatorsOffset);

            StringOffset subgraphNameOffset = fbb.CreateString(this.Name);

            Offset<tflite.Tensor>[] tensorsOffset = new Offset<tflite.Tensor>[Tensors.Length];
            for (int i = 0; i < tensorsOffset.Length; i++)
            {
                tensorsOffset[i] = Tensors[i].GetOffset(fbb);
            }

            VectorOffset tensorsOffsetVector = SubGraph.CreateTensorsVector(fbb, tensorsOffset);

            //subgraphs-inputs
            VectorOffset inputsOffsetVector = SubGraph.CreateInputsVector(fbb, this.Inputs);

            //subgraphs-inputs
            VectorOffset outputsOffsetVector = SubGraph.CreateOutputsVector(fbb, this.Outputs);

            return SubGraph.CreateSubGraph(fbb, tensorsOffsetVector, inputsOffsetVector, outputsOffsetVector, graphOperatorsOffsetVector, subgraphNameOffset);
        }
    }

    // 生データバッファのテーブル（定数テンソルに使用）
    // テンソルのインデックスで参照されます．
    // mmapフレンドリーなデータ構造に対応するため，余裕のある配置になっています．
    class Buffer
    {
        public byte[] Data;

        public Buffer(byte[] data = null)
        {
            float[] addVal = { 2, 6, 7 };
            byte[] addValByte = new byte[addVal.Length * sizeof(float)];
            System.Buffer.BlockCopy(addVal, 0, addValByte, 0, addValByte.Length);

            this.Data = data;
        }

        public Offset<tflite.Buffer> GetOffset(FlatBufferBuilder fbb)
        {
            return tflite.Buffer.CreateBuffer(fbb, this.Data == null ? default : tflite.Buffer.CreateDataVector(fbb, this.Data));
        }
    }

    class Metadata
    {
        // メタデータを一意に識別するための人が読める文字列
        public string Name;

        // バッファテーブルへのインデックス
        public uint Buffer;

        public Metadata(string name, uint buffer)
        {
            this.Name = name;
            this.Buffer = buffer;
        }

        public Offset<tflite.Metadata> GetOffset(FlatBufferBuilder fbb)
        {
            StringOffset name = fbb.CreateString(this.Name);
            return tflite.Metadata.CreateMetadata(fbb, name, this.Buffer);
        }
    }

    // テンソルのエイリアス名からグラフ内のテンソルインデックスへのマップ
    // これはSignatureのdefで使用されます
    class TensorMap
    {
        // このテンソルに使用するエイリアスを表します
        public string Name;

        // nameが対応する、プライマリグラフ内の実際のテンソルインデックス
        public uint tensor_index;
    }

    // これは、Tensorflow SavedModelのSignatureDefに相当します。
    // SignatureDefは変換用に提供されるSavedModelの一部になります。
    class SignatureDef
    {
        // Named inputs for this signature.
        // この署名のための命名された入力
        public List<TensorMap> Inputs = new List<TensorMap>();

        // Named outputs for this signature.
        // この署名のための命名された出力
        public List<TensorMap> Outputs = new List<TensorMap>();

        // Exported method name for this signature.
        // このシグネチャのエクスポートされたメソッド名
        public string MethodName;

        // Tensorflow SavedModel SignatureDefマップにあったキーの値
        public string key;
    }

    class Model
    {
        // スキーマのバージョン
        public uint Version;

        // このモデルで使用されているすべてのオペレーターコードのリストです。
        // 演算子はこのベクトルへのインデックスを持っているので、これは順番に保たれています。
        public List<OperatorCode> OperatorCodes = new List<OperatorCode>();

        // このモデルのすべてのサブグラフ。0番目をメインモデルとします。
        public List<Subgraph> Subgraphs = new List<Subgraph>();

        // モデルの説明
        public string Description = "Created by C#.";

        // モデルのバッファ
        // この配列の0番目のエントリは、空のバッファ(センチネル)でなければならないことに注意してください。
        // これはバッファを持たないテンソルが0をバッファとして提供できるようにするための規約です．
        public List<Buffer> Buffers = new List<Buffer>(new[] { new Buffer() });

        // モデルに関するメタデータ。存在するバッファリストへのインダイレクト。
        // 非推奨、メタデータフィールドの使用を推奨します。
        public List<int> MetadataBuffer = new List<int>();

        // モデルに関するメタデータ
        public List<Metadata> Metadata = new List<Metadata>();

        // オプションで、モデルのSignatureDefsがあります。
        public List<SignatureDef> SignatureDefs = new List<SignatureDef>();

        public byte[] Build()
        {
            //metadata
            byte[] runtimeVal = Encoding.ASCII.GetBytes("1.5.0");
            byte[] runtimeValByte = new byte[15];
            System.Buffer.BlockCopy(runtimeVal, 0, runtimeValByte, 0, runtimeVal.Length);
            Buffers.Add(new Schema.Buffer(runtimeValByte));

            Metadata.Add(new Schema.Metadata("min_runtime_version", 4));

            FlatBufferBuilder fbb = new FlatBufferBuilder(1);

            //operator_codes
            Offset<tflite.OperatorCode>[] operators = new Offset<tflite.OperatorCode>[OperatorCodes.Count];
            for (int i = 0; i < operators.Length; i++)
            {
                operators[i] = OperatorCodes[i].GetOffset(fbb);
            }
            VectorOffset operatorsOffset = tflite.Model.CreateOperatorCodesVector(fbb, operators);

            //subgraphs
            Offset<tflite.SubGraph>[] subgraphs = new Offset<tflite.SubGraph>[Subgraphs.Count];
            for (int i = 0; i < subgraphs.Length; i++)
            {
                subgraphs[i] = Subgraphs[i].GetOffset(fbb);
            }
            VectorOffset subgraphsOffsetVector = tflite.Model.CreateSubgraphsVector(fbb, subgraphs);

            //Description
            StringOffset descriptionOffset = fbb.CreateString(this.Description);

            //buffers
            Offset<tflite.Buffer>[] buffersOffset = new Offset<tflite.Buffer>[Buffers.Count];
            for (int i = 0; i < buffersOffset.Length; i++)
            {
                buffersOffset[i] = Buffers[i].GetOffset(fbb);
            }
            VectorOffset buffersOffsetVector = tflite.Model.CreateBuffersVector(fbb, buffersOffset);

            //MetaData
            Offset<tflite.Metadata>[] metadata = new Offset<tflite.Metadata>[Metadata.Count];
            for (int i = 0; i < metadata.Length; i++)
            {
                metadata[i] = Metadata[i].GetOffset(fbb);
            }
            VectorOffset metaDataoffset = tflite.Model.CreateMetadataVector(fbb, metadata);

            //signature_defs
            Offset<tflite.SignatureDef>[] signatureOffsets = new Offset<tflite.SignatureDef>[SignatureDefs.Count];
            //for(int i=0;i<signatureOffsets.Length;i++)
            //{
            //    signatureOffsets[i] = SignatureDefs.GetOffset(fbb);
            //}
            VectorOffset signatureDefsOffset = tflite.Model.CreateSignatureDefsVector(fbb, signatureOffsets);


            Offset<tflite.Model> model = tflite.Model.CreateModel(fbb, 3, operatorsOffset, subgraphsOffsetVector, descriptionOffset, buffersOffsetVector, default, metaDataoffset, signatureDefsOffset);
            tflite.Model.FinishModelBuffer(fbb, model);

            byte[] modelData = fbb.SizedByteArray();



            return modelData;
        }
    }
}
