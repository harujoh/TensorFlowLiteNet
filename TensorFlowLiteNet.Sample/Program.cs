using System;
using System.Linq;

namespace TensorFlowLiteNet.Sample
{
    class Program
    {
        static void Main(string[] args)
        {
            Variable<float> inputVar = new Variable<float>(1, 64, 64, 3);
            float[] inputConst = { 2.0f, 6.0f, 7.0f };
            float[] inputConst2 = { 1.0f, 1.0f, 1.0f };

            //計算グラフを定義
            var graph = inputVar + inputConst + inputConst2;

            //値を転送
            inputVar.SetVal(Enumerable.Range(0, inputVar.Length).Select(n => (float)n).ToArray());

            //保存
            //graph.Save("test.tfLite");

            //実行
            Variable<float> outputArray = graph.Run()[0];

            //結果を出力
            Console.WriteLine(outputArray);

            Console.Read();
        }
    }
}

