using System;
using System.Linq;
using System.Text;
using FlatBuffers;
using tflite;
using static TensorFlowLiteNet.NativeMethods;
using Buffer = tflite.Buffer;

namespace TensorFlowLiteNet.Sample
{
    class Program
    {
        static void Main(string[] args)
        {
            FlatBufferBuilder fbb = new FlatBufferBuilder(1);

            //operator_codes
            Offset<OperatorCode>[] operators = new Offset<OperatorCode>[1];
            operators[0] = OperatorCode.CreateOperatorCode(fbb, 0, version: 1, builtin_code: BuiltinOperator.ADD);
            VectorOffset operatorsOffset = Model.CreateOperatorCodesVector(fbb, operators);

            //subgraphs-tensors
            Offset<Tensor>[] tensorsOffset = new Offset<Tensor>[3];

            VectorOffset shapeOffset1Vector = Tensor.CreateShapeVector(fbb, new[] { 1, 64, 64, 3 });
            StringOffset nameOffset1 = fbb.CreateString("img");
            Offset<QuantizationParameters> quantizationOffset1Vector = QuantizationParameters.CreateQuantizationParameters(fbb, default, default, default, default, QuantizationDetails.NONE, 0, 0);
            tensorsOffset[0] = Tensor.CreateTensor(fbb, shapeOffset1Vector, TensorType.FLOAT32, 0, nameOffset1, quantizationOffset1Vector, false);

            VectorOffset shapeOffset2Vector = Tensor.CreateShapeVector(fbb, new[] { 3 });
            StringOffset nameOffset2 = fbb.CreateString("add");
            Offset<QuantizationParameters> quantizationOffset2Vector = QuantizationParameters.CreateQuantizationParameters(fbb, default, default, default, default, QuantizationDetails.NONE, 0, 0);
            tensorsOffset[1] = Tensor.CreateTensor(fbb, shapeOffset2Vector, TensorType.FLOAT32, 1, nameOffset2, quantizationOffset2Vector, false);

            VectorOffset shapeOffset3Vector = Tensor.CreateShapeVector(fbb, new[] { 1, 64, 64, 3 });
            StringOffset nameOffset3 = fbb.CreateString("out");
            Offset<QuantizationParameters> quantizationOffset3Vector = QuantizationParameters.CreateQuantizationParameters(fbb, default, default, default, default, QuantizationDetails.NONE, 0, 0);
            tensorsOffset[2] = Tensor.CreateTensor(fbb, shapeOffset3Vector, TensorType.FLOAT32, 2, nameOffset3, quantizationOffset3Vector, false);

            VectorOffset tensorsOffsetVector = SubGraph.CreateTensorsVector(fbb, tensorsOffset);

            //subgraphs-inputs
            VectorOffset inputsOffsetVector = SubGraph.CreateInputsVector(fbb, new[] { 0 });

            //subgraphs-inputs
            VectorOffset outputsOffsetVector = SubGraph.CreateOutputsVector(fbb, new[] { 2 });

            //subgraphs-operators-inputs
            VectorOffset operatorInputsOffsetVector = SubGraph.CreateInputsVector(fbb, new[] { 0, 1 });

            //subgraphs-operators-inputs
            VectorOffset operatorOutputsOffsetVector = SubGraph.CreateOutputsVector(fbb, new[] { 2 });

            //subgraphs-operators
            Offset<Operator>[] graphOperatorsOffset = new Offset<Operator>[1];
            //事前に用意しないとエラーが出る
            Offset<AddOptions> builtinOptionOffset = AddOptions.CreateAddOptions(fbb, ActivationFunctionType.NONE, true);

            //省略形ではbuiltinOptionが設定できない
            //graphOperatorsOffset[0] = Operator.CreateOperator(fbb, 0, operatorInputsOffsetVector, operatorOutputsOffsetVector, BuiltinOptions.AddOptions);
            Operator.StartOperator(fbb);
            Operator.AddOpcodeIndex(fbb, 0);
            Operator.AddInputs(fbb, operatorInputsOffsetVector);
            Operator.AddOutputs(fbb, operatorOutputsOffsetVector);
            Operator.AddBuiltinOptionsType(fbb, BuiltinOptions.AddOptions);
            Operator.AddBuiltinOptions(fbb, builtinOptionOffset.Value);
            Operator.AddCustomOptionsFormat(fbb, CustomOptionsFormat.FLEXBUFFERS);
            graphOperatorsOffset[0] = Operator.EndOperator(fbb);
            VectorOffset graphOperatorsOffsetVector = SubGraph.CreateOperatorsVector(fbb, graphOperatorsOffset);

            //subgraphs-name
            StringOffset subgraphNameOffset = fbb.CreateString("main");

            //subgraphs
            Offset<SubGraph>[] subgraphsOffset = new Offset<SubGraph>[1];
            subgraphsOffset[0] = SubGraph.CreateSubGraph(fbb, tensorsOffsetVector, inputsOffsetVector, outputsOffsetVector, graphOperatorsOffsetVector, subgraphNameOffset);
            VectorOffset subgraphsOffsetVector = Model.CreateSubgraphsVector(fbb, subgraphsOffset);

            //Description
            StringOffset descriptionOffset = fbb.CreateString("Created from C#.");

            //buffers
            Offset<Buffer>[] buffersOffset = new Offset<Buffer>[4];

            buffersOffset[0] = Buffer.CreateBuffer(fbb); //img

            float[] addVal = { 2, 6, 7 };
            byte[] addValByte = new byte[addVal.Length * sizeof(float)];
            System.Buffer.BlockCopy(addVal, 0, addValByte, 0, addValByte.Length);
            VectorOffset dataOffset2Vector = Buffer.CreateDataVector(fbb, addValByte);
            buffersOffset[1] = Buffer.CreateBuffer(fbb, dataOffset2Vector); //add

            buffersOffset[2] = Buffer.CreateBuffer(fbb); //out

            byte[] runtimeVal = Encoding.ASCII.GetBytes("1.5.0");
            byte[] runtimeValByte = new byte[15];
            System.Buffer.BlockCopy(runtimeVal, 0, runtimeValByte, 0, runtimeVal.Length);
            VectorOffset dataOffset4Vector = Buffer.CreateDataVector(fbb, runtimeValByte);
            buffersOffset[3] = Buffer.CreateBuffer(fbb, dataOffset4Vector); //metadata

            VectorOffset buffersOffsetVector = Model.CreateBuffersVector(fbb, buffersOffset);

            //MetaData
            Offset<Metadata>[] metadata = new Offset<Metadata>[1];
            StringOffset name = fbb.CreateString("min_runtime_version");
            metadata[0] = Metadata.CreateMetadata(fbb, name, 3);
            VectorOffset metaDataoffset = Model.CreateMetadataVector(fbb, metadata);

            //signature_defs
            Offset<SignatureDef>[] signatureOffsets = new Offset<SignatureDef>[0];
            VectorOffset signatureDefsOffset = Model.CreateSignatureDefsVector(fbb, signatureOffsets);

            //省略して書ける
            Offset<Model> model = Model.CreateModel(fbb, 3, operatorsOffset, subgraphsOffsetVector, descriptionOffset, buffersOffsetVector, default, metaDataoffset, signatureDefsOffset);
            //Offset<Model> model = Model.EndModel(fbb);
            //Model.StartModel(fbb);
            //Model.AddVersion(fbb, 3);
            //Model.AddOperatorCodes(fbb, operatorsOffset);
            //Model.AddSubgraphs(fbb, subgraphsOffsetVector);
            //Model.AddDescription(fbb, descriptionOffset);
            //Model.AddBuffers(fbb, buffersOffsetVector);
            //Model.AddMetadata(fbb, metaDataoffset);
            //Model.AddSignatureDefs(fbb,signatureDefsOffset);
            Model.FinishModelBuffer(fbb, model);


            //出来上がったtfliteをTensorFlowへ転送
            byte[] modelData = fbb.SizedByteArray();
            IntPtr tfmodel = TfLiteModelCreate(modelData, modelData.Length);

            IntPtr interpreter = TfLiteInterpreterCreate(tfmodel, IntPtr.Zero);

            TfLiteInterpreterAllocateTensors(interpreter);

            //モデルから入力のサイズを取得し作成
            int inputCount = TfLiteInterpreterGetInputTensorCount(interpreter);
            IntPtr inputTensorPtr = TfLiteInterpreterGetInputTensor(interpreter, 0);
            int[] inputDim = new int[TfLiteTensorNumDims(inputTensorPtr)];
            for (int i = 0; i < inputDim.Length; i++)
            {
                inputDim[i] = TfLiteTensorDim(inputTensorPtr, i);
            }

            //入力に適当な値を代入
            NdArray<float> inputArray = new NdArray<float>(inputDim);
            inputArray.Data = Enumerable.Range(0, inputArray.Length).Select(n => (float)n).ToArray();

            //TFに転送して
            TfLiteTensorCopyFromBuffer(inputTensorPtr, inputArray.Data);

            //実行
            TfLiteInterpreterInvoke(interpreter);

            //モデルから出力のサイズを取得し作成
            int outputCount = TfLiteInterpreterGetOutputTensorCount(interpreter);
            IntPtr outputTensorPtr = TfLiteInterpreterGetOutputTensor(interpreter, 0);
            int[] outputDim = new int[TfLiteTensorNumDims(outputTensorPtr)];
            for (int i = 0; i < outputDim.Length; i++)
            {
                outputDim[i] = TfLiteTensorDim(outputTensorPtr, i);
            }

            //結果の取得
            NdArray<float> outputArray = new NdArray<float>(outputDim);
            TfLiteTensorCopyToBuffer(outputTensorPtr, outputArray.Data);

            //結果を出力
            Console.WriteLine(outputArray);

            //後片付け
            TfLiteInterpreterDelete(interpreter);
            interpreter = IntPtr.Zero;

            TfLiteModelDelete(tfmodel);
            tfmodel = IntPtr.Zero;


            Console.Read();
        }
    }
}
