using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorFlowLiteNet
{
    [DebuggerDisplay("{\"NdArray\" + ShapeString()}", Type = "{\"NdArray\" + ShapeString()}")]
    public struct NdArray<T>
    {
        public T[] Data;

        //このNdArrayの各次元のサイズ
        public int[] Shape;

        public int Length => this.Data.Length;

        public NdArray(Array array)
        {
            this.Shape = new int[array.Rank];
            this.Data = new T[array.Length];

            for (int i = 0; i < this.Shape.Length; i++)
            {
                this.Shape[i] = array.GetLength(i);
            }

            Buffer.BlockCopy(array, 0, this.Data, 0, this.Data.Length * Marshal.SizeOf<T>());
        }

        public NdArray(params int[] shape)
        {
            this.Data = new T[NdArray.ShapeToLength(shape)];
            this.Shape = new int[shape.Length];

            Buffer.BlockCopy(shape, 0, this.Shape, 0, this.Shape.Length * sizeof(int));
        }

        //インデクサはあまり早くないので頻繁にアクセスする場合は使用をオススメしません。デバッグ用途向けと割り切ってください。
        public T this[params int[] indices]
        {
            get => this.Data[NdArray.GetLocalIndex(this.Shape, indices)];
            set => this.Data[NdArray.GetLocalIndex(this.Shape, indices)] = value;
        }

        public override string ToString()
        {
            return NdArray.ToString(this.Data, this.Shape);
        }

        string ShapeString()
        {
            return "[" + string.Join(",", this.Shape) + "]";
        }
    }

    static class NdArray
    {
        public static int GetLocalIndex(int[] shape, params int[] indices)
        {
            int result = 0;
            int rankOffset = 1;

            for (int i = indices.Length - 1; i >= 0; i--)
            {
                result += indices[i] * rankOffset;
                rankOffset *= shape[i];
            }

            return result;
        }

        public static int ShapeToLength(params int[] shapes)
        {
            int result = 1;

            foreach (int shape in shapes)
            {
                result *= shape;
            }

            return result;
        }

        public static string ToString<T>(T[] arrayData, int[] shape)
        {
#if DEBUG
            if (shape.Length == 0) throw new Exception();
#endif
            if (arrayData.Length < 2)
            {
                return arrayData[0].ToString();
            }

            StringBuilder sb = new StringBuilder();

            int intMaxLength = 0; //整数部の最大値
            int realMaxLength = 0; //小数点以下の最大値
            bool isExponential = false; //指数表現にするか

            foreach (T data in arrayData)
            {
                string[] divStr = data.ToString()?.Split('.') ?? new[] { "" };
                intMaxLength = (int)Math.Max(intMaxLength, divStr[0].Length);

                if (divStr.Length > 1)
                {
                    isExponential |= divStr[1].Contains("E");
                }

                if (realMaxLength != 8 && divStr.Length == 2)
                {
                    realMaxLength = (int)Math.Max(realMaxLength, divStr[1].Length);

                    if (realMaxLength > 8)
                    {
                        realMaxLength = 8;
                    }
                }
            }

            //配列の約数を取得
            int[] commonDivisorList = new int[shape.Length];

            //一個目は手動取得
            commonDivisorList[0] = shape[shape.Length - 1];

            for (int i = 1; i < shape.Length; i++)
            {
                commonDivisorList[i] = commonDivisorList[i - 1] * shape[shape.Length - i - 1];
            }

            //先頭の括弧
            for (int i = 0; i < shape.Length; i++)
            {
                sb.Append("[");
            }

            int closer = 0;
            for (int i = 0; i < arrayData.Length; i++)
            {
                string[] divStr;
                if (isExponential)
                {
                    divStr = string.Format("{0:0.00000000e+00}", arrayData[i]).Split('.');
                }
                else
                {
                    divStr = arrayData[i].ToString()?.Split('.') ?? new[] { "" };
                }

                //最大文字数でインデントを揃える
                for (int j = 0; j < intMaxLength - divStr[0].Length; j++)
                {
                    sb.Append(" ");
                }

                sb.Append(divStr[0]);

                if (realMaxLength != 0)
                {
                    sb.Append(".");
                }

                if (divStr.Length == 2)
                {
                    sb.Append(divStr[1].Length > 8 && !isExponential ? divStr[1].Substring(0, 8) : divStr[1]);
                    for (int j = 0; j < realMaxLength - divStr[1].Length; j++)
                    {
                        sb.Append(" ");
                    }
                }
                else
                {
                    for (int j = 0; j < realMaxLength; j++)
                    {
                        sb.Append(" ");
                    }
                }

                //約数を調査してピッタリなら括弧を出力
                if (i != arrayData.Length - 1)
                {
                    foreach (int commonDivisor in commonDivisorList)
                    {
                        if ((i + 1) % commonDivisor == 0)
                        {
                            sb.Append("]");
                            closer++;
                        }
                    }

                    sb.Append(" ");

                    if ((i + 1) % commonDivisorList[0] == 0)
                    {
                        //閉じ括弧分だけ改行を追加
                        for (int j = 0; j < closer; j++)
                        {
                            sb.Append("\n");
                        }
                        closer = 0;

                        //括弧前のインデント
                        foreach (int commonDivisor in commonDivisorList)
                        {
                            if ((i + 1) % commonDivisor != 0)
                            {
                                sb.Append(" ");
                            }
                        }
                    }

                    foreach (int commonDivisor in commonDivisorList)
                    {
                        if ((i + 1) % commonDivisor == 0)
                        {
                            sb.Append("[");
                        }
                    }
                }
            }

            //後端の括弧
            for (int i = 0; i < shape.Length; i++)
            {
                sb.Append("]");
            }

            return sb.ToString();
        }
    }
}
