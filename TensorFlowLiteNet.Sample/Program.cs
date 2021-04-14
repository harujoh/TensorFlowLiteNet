using System;
using System.Linq;

namespace TensorFlowLiteNet.Sample
{
    class Program
    {
        static void Main(string[] args)
        {
            Variable<float> inputVar = new Variable<float>(1, 64, 64, 3);
            float[] inputConst = { 0.5f, 0.5f, 0.5f };
            float[] inputConst2 = { 2.0f, 2.0f, 2.0f };
            float[] inputConst3 = { 4.0f, 4.0f, 4.0f };
            float[] inputConst4 = { 3.0f, 3.0f, 3.0f };

            //計算グラフを定義
            var graph = inputVar / inputConst * inputConst2 + inputConst3 - inputConst4;

            //保存
            //graph.Save("test.tfLite");

            //実行
            Variable<float> outputArray = graph.Predict(Enumerable.Range(0, inputVar.Length).Select(n => (float) n).ToArray())[0];

            //結果を出力
            Console.WriteLine(outputArray);

            Console.Read();
        }
    }
}

