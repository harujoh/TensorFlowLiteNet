using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;

namespace TensorFlowLiteNet
{
    [DebuggerDisplay("{\"NdArray\" + ShapeString()}", Type = "{\"NdArray\" + ShapeString()}")]
    public class Variable<T>
    {
        public T[] Data;
        public int[] Shape;
        public string Name;

        public int Length => this.Data.Length;

        public Variable(Array array, string name = "")
        {
            this.Shape = new int[array.Rank];
            this.Data = new T[array.Length];

            for (int i = 0; i < this.Shape.Length; i++)
            {
                this.Shape[i] = array.GetLength(i);
            }

            Buffer.BlockCopy(array, 0, this.Data, 0, this.Data.Length * Unsafe.SizeOf<T>());

            this.Name = name;
        }

        public Variable(params int[] shape)
        {
            this.Data = new T[NdArray.ShapeToLength(shape)];
            this.Shape = new int[shape.Length];

            Buffer.BlockCopy(shape, 0, this.Shape, 0, this.Shape.Length * sizeof(int));

            this.Name = "";
        }

        public Variable(T[] data, int[] shape, string name = "")
        {
#if DEBUG
            if (data.Length != NdArray.ShapeToLength(shape)) throw new Exception("指定された配列とシェイプが一致していません");
#endif
            this.Data = new T[data.Length];
            Buffer.BlockCopy(data, 0, this.Data, 0, Unsafe.SizeOf<T>() * data.Length);
            this.Shape = new int[shape.Length];
            Buffer.BlockCopy(shape, 0, this.Shape, 0, sizeof(int) * shape.Length);

            this.Name = name;
        }

        //インデクサはあまり早くないので頻繁にアクセスする場合は使用をオススメしません。
        //デバッグ用途か結果のプレビュー向けと割り切ってください。
        public T this[params int[] indices]
        {
            get => this.Data[NdArray.GetLocalIndex(this.Shape, indices)];
            set => this.Data[NdArray.GetLocalIndex(this.Shape, indices)] = value;
        }

        public byte[] GetBytes()
        {
            byte[] result = new byte[this.Data.Length * Unsafe.SizeOf<T>()];
            Buffer.BlockCopy(this.Data, 0, result, 0, result.Length);
            return result;
        }

        public void SetVal(Array data)
        {
            Buffer.BlockCopy(data, 0, this.Data, 0, this.Data.Length * Unsafe.SizeOf<T>());
        }

        public static Graph<T> operator +(Variable<T> a, Array b)
        {
            return new Graph<T>(a) + b;
        }

        public static Graph<T> operator -(Variable<T> a, Array b)
        {
            return new Graph<T>(a) - b;
        }

        public static Graph<T> operator *(Variable<T> a, Array b)
        {
            return new Graph<T>(a) * b;
        }

        public static Graph<T> operator /(Variable<T> a, Array b)
        {
            return new Graph<T>(a) / b;
        }

        string ShapeString()
        {
            return "[" + string.Join(",", this.Shape) + "]";
        }

        public override string ToString()
        {
            return NdArray.ToString(this.Data, this.Shape);
        }
    }

    public static class NdArray
    {
        public static int[] Broadcast(int[] source, int[] target)
        {
            int[] resultShape;

            if (source.Length > target.Length)
            {
                //入力の方が大きい
                resultShape = new int[source.Length];//val.ToArray();
                int offset = source.Length - target.Length;

                Buffer.BlockCopy(source, 0, resultShape, 0, offset * sizeof(int));
                for (int i = offset; i < resultShape.Length; i++)
                {
                    resultShape[i] = source[i];

                    if (resultShape[i] == 1)
                    {
                        resultShape[i] = target[i - offset];
                    }
#if DEBUG
                    else if (target[i - offset] != 1 && resultShape[i] != target[i - offset])
                    {
                        throw new Exception("変換不可能な組み合わせです");
                    }
#endif
                }
            }
            else
            {
                //指定の方が大きい
                resultShape = new int[target.Length];//Shape.ToArray();
                int offset = target.Length - source.Length;

                Buffer.BlockCopy(target, 0, resultShape, 0, offset * sizeof(int));
                for (int i = offset; i < resultShape.Length; i++)
                {
                    resultShape[i] = target[i];

                    if (resultShape[i] == 1)
                    {
                        resultShape[i] = source[i - offset];
                    }
#if DEBUG
                    else if (source[i - offset] != 1 && resultShape[i] != source[i - offset])
                    {
                        throw new Exception("変換不可能な組み合わせです");
                    }
#endif
                }
            }

            return resultShape;
        }

        public static int[] GetDimensionsIndex(int[] shape, int index)
        {
            int[] dimensionsIndex = new int[shape.Length];

            for (int i = shape.Length - 1; i >= 0; i--)
            {
                dimensionsIndex[i] = index % shape[i];
                index /= shape[i];
            }

            return dimensionsIndex;
        }

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
