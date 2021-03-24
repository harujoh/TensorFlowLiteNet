using System;
using System.Runtime.InteropServices;

using TfLiteInterpreter = System.IntPtr;
using TfLiteInterpreterOptions = System.IntPtr;
using TfLiteModel = System.IntPtr;
using TfLiteTensor = System.IntPtr;
using TfLiteDelegate = System.IntPtr;

namespace TensorFlowLiteNet
{
    public enum edgetpu_device_type
    {
        EDGETPU_APEX_PCI = 0,
        EDGETPU_APEX_USB = 1,
    };

    public struct edgetpu_device
    {
        edgetpu_device_type type;
        string path;
    };

    public struct edgetpu_option
    {
        string name;
        string value;
    };

    public enum DataType
    {
        NoType = 0,
        Float32 = 1,
        Int32 = 2,
        UInt8 = 3,
        Int64 = 4,
        String = 5,
        Bool = 6,
        Int16 = 7,
        Complex64 = 8,
        Int8 = 9,
        Float16 = 10,
    }

    public struct QuantizationParams
    {
        float scale;
        int zeroPoint;
    }

    public static class NativeMethods
    {
        [DllImport("edgetpu.dll")]
        public static extern IntPtr edgetpu_version();

        [DllImport("edgetpu.dll")]
        public static extern edgetpu_device edgetpu_list_devices(UIntPtr num_devices);

        [DllImport("edgetpu.dll")]
        public static extern void edgetpu_free_devices(edgetpu_device dev);

        [DllImport("edgetpu.dll")]
        public static extern TfLiteDelegate edgetpu_create_delegate(edgetpu_device_type type, [In] string name, [In] edgetpu_option options, UIntPtr num_options);

        [DllImport("edgetpu.dll")]
        public static extern void edgetpu_free_delegate(TfLiteDelegate tfDelegate);

        [DllImport("edgetpu.dll")]
        public static extern void edgetpu_verbosity(int verbosity);

        public static string GetEdgeTPUVersion()
        {
            return Marshal.PtrToStringAnsi(edgetpu_version());
        }


        [DllImport("tensorflowlite_c.dll")]
        public static extern IntPtr TfLiteVersion();

        [DllImport("tensorflowlite_c.dll")]
        public static extern TfLiteInterpreter TfLiteModelCreate(byte[] model_data, int model_size);

        [DllImport("tensorflowlite_c.dll")]
        public static extern TfLiteInterpreter TfLiteModelDelete(TfLiteModel model);

        [DllImport("tensorflowlite_c.dll")]
        public static extern TfLiteInterpreterOptions TfLiteInterpreterOptionsCreate();

        [DllImport("tensorflowlite_c.dll")]
        public static extern void TfLiteInterpreterOptionsDelete(TfLiteInterpreterOptions options);

        [DllImport("tensorflowlite_c.dll")]
        public static extern void TfLiteInterpreterOptionsSetNumThreads(TfLiteInterpreterOptions options, int num_threads);

        [DllImport("tensorflowlite_c.dll")]
        public static extern TfLiteInterpreter TfLiteInterpreterCreate(TfLiteModel model, TfLiteInterpreterOptions optional_options);

        [DllImport("tensorflowlite_c.dll")]
        public static extern void TfLiteInterpreterDelete(TfLiteInterpreter interpreter);

        [DllImport("tensorflowlite_c.dll")]
        public static extern int TfLiteInterpreterGetInputTensorCount(TfLiteInterpreter interpreter);

        [DllImport("tensorflowlite_c.dll")]
        public static extern TfLiteTensor TfLiteInterpreterGetInputTensor(TfLiteInterpreter interpreter, int input_index);

        [DllImport("tensorflowlite_c.dll")]
        public static extern int TfLiteInterpreterResizeInputTensor(TfLiteInterpreter interpreter, int input_index, int[] input_dims, int input_dims_size);

        [DllImport("tensorflowlite_c.dll")]
        public static extern int TfLiteInterpreterAllocateTensors(TfLiteInterpreter interpreter);

        [DllImport("tensorflowlite_c.dll")]
        public static extern int TfLiteInterpreterInvoke(TfLiteInterpreter interpreter);

        [DllImport("tensorflowlite_c.dll")]
        public static extern int TfLiteInterpreterGetOutputTensorCount(TfLiteInterpreter interpreter);

        [DllImport("tensorflowlite_c.dll")]
        public static extern TfLiteTensor TfLiteInterpreterGetOutputTensor(TfLiteInterpreter interpreter, int output_index);

        [DllImport("tensorflowlite_c.dll")]
        public static extern DataType TfLiteTensorType(TfLiteTensor tensor);

        [DllImport("tensorflowlite_c.dll")]
        public static extern int TfLiteTensorNumDims(TfLiteTensor tensor);

        [DllImport("tensorflowlite_c.dll")]
        public static extern int TfLiteTensorDim(TfLiteTensor tensor, int dim_index);

        [DllImport("tensorflowlite_c.dll")]
        public static extern uint TfLiteTensorByteSize(TfLiteTensor tensor);

        [DllImport("tensorflowlite_c.dll")]
        public static extern IntPtr TfLiteTensorName(TfLiteTensor tensor);

        [DllImport("tensorflowlite_c.dll")]
        public static extern QuantizationParams TfLiteTensorQuantizationParams(TfLiteTensor tensor);

        [DllImport("tensorflowlite_c.dll")]
        static extern int TfLiteTensorCopyFromBuffer(TfLiteTensor tensor, IntPtr input_data, int input_data_size);

        public static int TfLiteTensorCopyFromBuffer<T>(TfLiteTensor tensor, T[] input_data)
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(input_data, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();

            return TfLiteTensorCopyFromBuffer(tensor, tensorDataPtr, input_data.Length * Marshal.SizeOf<T>());
        }


        [DllImport("tensorflowlite_c.dll")]
        static extern int TfLiteTensorCopyToBuffer(TfLiteTensor tensor, IntPtr output_data, int output_data_size);
        
        public static int TfLiteTensorCopyToBuffer<T>(TfLiteTensor tensor, T[] output_data)
        {
            GCHandle tensorDataHandle = GCHandle.Alloc(output_data, GCHandleType.Pinned);
            IntPtr tensorDataPtr = tensorDataHandle.AddrOfPinnedObject();

            return TfLiteTensorCopyToBuffer(tensor, tensorDataPtr, output_data.Length * Marshal.SizeOf<T>());
        }


        [DllImport("tensorflowlite_c.dll")]
        public static extern void TfLiteInterpreterOptionsAddDelegate(TfLiteInterpreterOptions options, TfLiteDelegate delegate_);


        public static string GetTFLiteVersion()
        {
            return Marshal.PtrToStringAnsi(TfLiteVersion());
        }
    }
}
