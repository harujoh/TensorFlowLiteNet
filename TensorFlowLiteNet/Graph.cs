using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using tflite;
using static TensorFlowLiteNet.NativeMethods;

namespace TensorFlowLiteNet
{
    public class Graph<T>
    {
        Schema.Model schemaModel = new Schema.Model();

        private List<Schema.Tensor> tensorsOffset = new List<Schema.Tensor>();
        private List<Schema.Operator> operatorsOffset = new List<Schema.Operator>();

        private Dictionary<BuiltinOperator, uint> OperatorDict = new Dictionary<BuiltinOperator, uint>();

        private Variable<T>[] inputArrays;
        private Variable<T>[] outputArrays;

        private int[] InputsTensorIndex = null;
        private int[] OutputsTensorIndex = null;

        TensorType GetTensorType()
        {
            if (typeof(T) == typeof(int)) return TensorType.INT32;
            if (typeof(T) == typeof(long)) return TensorType.INT64;
            if (typeof(T) == typeof(float)) return TensorType.FLOAT32;
            if (typeof(T) == typeof(double)) return TensorType.FLOAT64;
            throw new Exception("サポートされていない型です");
        }

        public Graph(Variable<T> inputVar)
        {
            TensorType tensorType = GetTensorType();

            InputsTensorIndex = new[] { 0 }; //PlusConstは右辺が定数なので左辺のみ
            inputArrays = new[] { inputVar };

            tensorsOffset.Add(new Schema.Tensor(inputVar.Shape, tensorType, (uint)schemaModel.Buffers.Count, schemaModel.Buffers.Count + ":" + inputVar.Name));
            schemaModel.Buffers.Add(new Schema.Buffer());

            outputArrays = new[] { new Variable<T>(inputVar.Shape) };
            OutputsTensorIndex = new[] { 0 }; //特に計算は行わないので自分自身を返す
        }

        private void AddConstOperator(Array input, BuiltinOperator builtinOperator, Schema.IBuiltinOptions builtinOption)
        {
            List<int> inputs = new List<int>();

            if (!OperatorDict.ContainsKey(builtinOperator))
            {
                OperatorDict.Add(builtinOperator, (uint)operatorsOffset.Count);
                schemaModel.OperatorCodes.Add(new Schema.OperatorCode(0, version: 1, builtin_code: builtinOperator));
            }

            TensorType tensorType = GetTensorType();

            //前の出力を入力に
            inputs.AddRange(OutputsTensorIndex.ToArray());

            Variable<T> inputConst = new Variable<T>(input);

            inputs.Add(tensorsOffset.Count);
            tensorsOffset.Add(new Schema.Tensor(inputConst.Shape, tensorType, (uint)schemaModel.Buffers.Count, schemaModel.Buffers.Count + ":" + inputConst.Name));//Tensorに追加時はbufferのIndex
            schemaModel.Buffers.Add(new Schema.Buffer(inputConst.GetBytes()));

            int[] result = outputArrays[0].Shape;//outputArraysが入力になるので
            if (outputArrays[0].Length != inputConst.Length)
            {
                result = outputArrays[0].Length < inputConst.Length ?
                    NdArray.Broadcast(outputArrays[0].Shape, inputConst.Shape) :
                    NdArray.Broadcast(inputConst.Shape, outputArrays[0].Shape);
            }

            //計算グラフの出力シェイプを更新
            outputArrays = new[] { new Variable<T>(result) };

            int[] outputs = { tensorsOffset.Count };
            tensorsOffset.Add(new Schema.Tensor(result, tensorType, (uint)schemaModel.Buffers.Count, schemaModel.Buffers.Count + ":" + "out"));
            schemaModel.Buffers.Add(new Schema.Buffer());

            OutputsTensorIndex = outputs;

            //opの追加時はTensorのIndex
            operatorsOffset.Add(new Schema.Operator(OperatorDict[builtinOperator], inputs.ToArray(), outputs, builtinOption));
        }

        public void Save(string path)
        {
            schemaModel.Subgraphs.Add(new Schema.Subgraph(tensorsOffset.ToArray(), InputsTensorIndex, OutputsTensorIndex, operatorsOffset.ToArray(), "main"));

            byte[] modelData = schemaModel.Build();

            //保存
            using (FileStream fs = new FileStream(path, FileMode.Create))
            {
                using (BinaryWriter bw = new BinaryWriter(fs))
                {
                    bw.Write(modelData);
                }
            }
        }

        public Variable<T>[] Predict(params Array[] arrays)
        {
            for (int i = 0; i < arrays.Length; i++)
            {
                inputArrays[i].SetVal(arrays[i]);
            }

            return Run();
        }

        public Variable<T>[] Run()
        {
            schemaModel.Subgraphs.Add(new Schema.Subgraph(tensorsOffset.ToArray(), InputsTensorIndex, OutputsTensorIndex, operatorsOffset.ToArray(), "main"));

            byte[] modelData = schemaModel.Build();

            IntPtr tfmodel = TfLiteModelCreate(modelData, modelData.Length);

            IntPtr interpreter = TfLiteInterpreterCreate(tfmodel, IntPtr.Zero);

            TfLiteInterpreterAllocateTensors(interpreter);

            //モデルから入力のサイズを取得し作成
            int inputCount = TfLiteInterpreterGetInputTensorCount(interpreter);

            for (int i = 0; i < inputCount; i++)
            {
#if DEBUG
                IntPtr inputTensorPtr = TfLiteInterpreterGetInputTensor(interpreter, i);
                int[] inputDim = new int[TfLiteTensorNumDims(inputTensorPtr)];
                if (inputArrays[i].Shape.Length != inputDim.Length) throw new Exception("入力されたサイズとモデルの入力次元が異なります");
                for (int j = 0; j < inputDim.Length; j++)
                {
                    inputDim[j] = TfLiteTensorDim(inputTensorPtr, j);
                    if (inputArrays[i].Shape[j] != inputDim[j]) throw new Exception("入力されたサイズとモデルの入力サイズが異なります");
                }
#endif
                //TFに転送して
                TfLiteTensorCopyFromBuffer(inputTensorPtr, inputArrays[i].Data);
            }

            //実行
            TfLiteInterpreterInvoke(interpreter);

            //モデルから出力のサイズを取得し作成
            int outputCount = TfLiteInterpreterGetOutputTensorCount(interpreter);
            Variable<T>[] outputArrays = new Variable<T>[outputCount];

            for (int i = 0; i < outputArrays.Length; i++)
            {
                IntPtr outputTensorPtr = TfLiteInterpreterGetOutputTensor(interpreter, i);
                int[] outputDim = new int[TfLiteTensorNumDims(outputTensorPtr)];

                for (int j = 0; j < outputDim.Length; j++)
                {
                    outputDim[j] = TfLiteTensorDim(outputTensorPtr, j);
                }

                //結果の取得
                outputArrays[i] = new Variable<T>(outputDim);
                TfLiteTensorCopyToBuffer(outputTensorPtr, outputArrays[i].Data);
            }

            //後片付け
            TfLiteInterpreterDelete(interpreter);
            interpreter = IntPtr.Zero;

            TfLiteModelDelete(tfmodel);
            tfmodel = IntPtr.Zero;

            return outputArrays;
        }

        public static Graph<T> operator +(Graph<T> a, Array b)
        {
            Schema.AddOptions builtinOption = new Schema.AddOptions(ActivationFunctionType.NONE, true);

            a.AddConstOperator(b, BuiltinOperator.ADD, builtinOption);
            return a;
        }

        public static Graph<T> operator -(Graph<T> a, Array b)
        {
            Schema.SubOptions builtinOption = new Schema.SubOptions(ActivationFunctionType.NONE, true);

            a.AddConstOperator(b, BuiltinOperator.SUB, builtinOption);
            return a;
        }

        public static Graph<T> operator *(Graph<T> a, Array b)
        {
            Schema.MulOptions builtinOption = new Schema.MulOptions(ActivationFunctionType.NONE);

            a.AddConstOperator(b, BuiltinOperator.MUL, builtinOption);
            return a;
        }

        public static Graph<T> operator /(Graph<T> a, Array b)
        {
            Schema.DivOptions builtinOption = new Schema.DivOptions(ActivationFunctionType.NONE);

            a.AddConstOperator(b, BuiltinOperator.DIV, builtinOption);
            return a;
        }
    }
}
