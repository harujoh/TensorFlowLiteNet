using System;
using System.Collections.Generic;
using System.IO;
using tflite;
using static TensorFlowLiteNet.NativeMethods;

namespace TensorFlowLiteNet
{
    public class ModelBuilder<T>
    {
        Schema.Model schemaModel = new Schema.Model();

        private List<Schema.Tensor> tensorsOffset = new List<Schema.Tensor>();
        private List<Schema.Operator> operatorsOffset = new List<Schema.Operator>();

        private Dictionary<BuiltinOperator, uint> OperatorDict = new Dictionary<BuiltinOperator, uint>();

        private Variable<T>[] inputArrays;

        private int[] InputsTensorIndex = null;
        private int[] OutputsTensorIndex = null;

        public void AddPlusConstOperator(Variable<T> inputVar, Array input)
        {
            List<int> inputs = new List<int>();
            List<int> outputs = new List<int>();

            if (!OperatorDict.ContainsKey(BuiltinOperator.ADD))
            {
                OperatorDict.Add(BuiltinOperator.ADD, (uint)operatorsOffset.Count);
                schemaModel.OperatorCodes.Add(new Schema.OperatorCode(0, version: 1, builtin_code: BuiltinOperator.ADD));
            }

            TensorType tensorType = TensorType.INT32;
            if (typeof(T) == typeof(int)) tensorType = TensorType.INT32;
            if (typeof(T) == typeof(long)) tensorType = TensorType.INT64;
            if (typeof(T) == typeof(float)) tensorType = TensorType.FLOAT32;
            if (typeof(T) == typeof(double)) tensorType = TensorType.FLOAT64;

            if (InputsTensorIndex == null) {
                InputsTensorIndex = new[] { 0 };//PlusConstは右辺が定数なので左辺のみ
                inputArrays = new[] { inputVar };
            }


            //img
            inputs.Add(tensorsOffset.Count);
            tensorsOffset.Add(new Schema.Tensor(inputVar.Shape, tensorType, (uint)schemaModel.Buffers.Count, schemaModel.Buffers.Count + ":" + inputVar.Name));//Tensorに追加時はbufferのIndex
            schemaModel.Buffers.Add(new Schema.Buffer());

            //add
            Variable<T> inputConst = new Variable<T>(input);

            inputs.Add(tensorsOffset.Count);
            tensorsOffset.Add(new Schema.Tensor(inputConst.Shape, tensorType, (uint)schemaModel.Buffers.Count, schemaModel.Buffers.Count + ":" + inputConst.Name));
            schemaModel.Buffers.Add(new Schema.Buffer(inputConst.GetBytes()));

            int[] result = inputVar.Shape;
            if (inputVar.Length != inputConst.Length)
            {
                result = inputVar.Length < inputConst.Length?
                    NdArray.Broadcast(inputVar.Shape, inputConst.Shape):
                    NdArray.Broadcast(inputConst.Shape, inputVar.Shape);
            }

            //out
            outputs.Add(tensorsOffset.Count);
            tensorsOffset.Add(new Schema.Tensor(result, tensorType, (uint)schemaModel.Buffers.Count, schemaModel.Buffers.Count + ":" + "out"));
            schemaModel.Buffers.Add(new Schema.Buffer());

            OutputsTensorIndex = outputs.ToArray();

            //事前に用意しないとエラーが出る
            Schema.AddOptions builtinOption = new Schema.AddOptions(ActivationFunctionType.NONE, true);

            //opの追加時はTensorのIndex
            operatorsOffset.Add(new Schema.Operator(OperatorDict[BuiltinOperator.ADD], inputs.ToArray(), outputs.ToArray(), BuiltinOptions.AddOptions, builtinOption));
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
    }
}
