using System;
using System.Runtime.InteropServices;

using static TensorFlowLiteNet.NativeMethods;

namespace TensorFlowLiteNet.Sample
{
    class Program
    {
        static string GetTFLiteVersion()
        {
            return Marshal.PtrToStringAnsi(TfLiteVersion());
        }

        static string GetEdgeTPUVersion()
        {
            return Marshal.PtrToStringAnsi(edgetpu_version());
        }

        static void Main(string[] args)
        {
            Console.WriteLine(GetTFLiteVersion());
            Console.WriteLine(GetEdgeTPUVersion());

            Console.Read();
        }
    }
}
