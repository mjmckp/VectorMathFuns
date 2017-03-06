using System;
using System.Linq;
using System.Numerics;

namespace MathFun
{
    /// <summary>
    /// Dummy implementations of AVX intrinsics required
    /// https://software.intel.com/sites/landingpage/IntrinsicsGuide/
    /// </summary>
    public static class AVXIntrinsics
    {
        /// <summary>
        /// Floor
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Vector<float> _mm256_floor_ps(Vector<float> x)
        {
            var rslt = new float[Vector<float>.Count];
            for (int i = 0; i < rslt.Length; ++i)
                rslt[i] = (float) Math.Floor(x[i]);
            return new Vector<float>(rslt);
        }

        /// <summary>
        /// Conversion from float to int
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Vector<int> _mm256_cvttps_epi32(Vector<float> x)
        {
            var rslt = new int[Vector<float>.Count];
            for (int i = 0; i < rslt.Length; ++i)
                rslt[i] = (int) x[i];
            return new Vector<int>(rslt);
        }

        /// <summary>
        /// Conversion from int to float
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static Vector<float> _mm256_cvtepi32_ps(Vector<int> x)
        {
            var rslt = new float[Vector<int>.Count];
            for (int i = 0; i < rslt.Length; ++i)
                rslt[i] = (float)x[i];
            return new Vector<float>(rslt);
        }

        /// <summary>
        /// Bitwise shift-left
        /// </summary>
        /// <param name="x"></param>
        /// <param name="imm8"></param>
        /// <returns></returns>
        public static Vector<int> _mm256_slli_epi32(Vector<int> x, int imm8)
        {
            var rslt = new int[Vector<int>.Count];
            for (int i = 0; i < rslt.Length; ++i)
                rslt[i] = x[i] << imm8;
            return new Vector<int>(rslt);
        }

        /// <summary>
        /// Bitwise shift-right
        /// </summary>
        /// <param name="x"></param>
        /// <param name="imm8"></param>
        /// <returns></returns>
        public static Vector<int> _mm256_srli_epi32(Vector<int> x, int imm8)
        {
            var rslt = new int[Vector<float>.Count];
            for (int i = 0; i < rslt.Length; ++i)
                rslt[i] = x[i] >> imm8;
            return new Vector<int>(rslt);
        }
                
        /// <summary>
        /// Bitwise XOR
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        // TODO: works for any Vector<T>
        public static Vector<int> _mm256_xor_si256(Vector<int> x, Vector<int> y)
        {
            var rslt = new int[Vector<int>.Count];
            for (int i = 0; i < rslt.Length; ++i)
                rslt[i] = x[i] ^ y[i];
            return new Vector<int>(rslt);
        }

        /// <summary>
        /// Compute the bitwise NOT of packed single-precision (32-bit) 
        /// floating-point elements in x and then AND with y.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public static Vector<int> _mm256_andnot_si256(Vector<int> x, Vector<int> y)
        {
            var rslt = new int[Vector<int>.Count];
            for (int i = 0; i < rslt.Length; ++i)
                rslt[i] = (~x[i]) & y[i];
            return new Vector<int>(rslt);
        }

    }

    public static class VectorMathFun
    {
        static Vector<float> one = Vector<float>.One;
        static Vector<float> zero = Vector<float>.Zero;
        static Vector<float> nan = new Vector<float>(float.NaN);
        static Vector<float> _ps256_0p5 = new Vector<float>(0.5f);
        static Vector<int> _pi32_256_0x7f = new Vector<int>(0x7f);
        static Vector<int> _pi32_256_inv1 = new Vector<int>(~1);
        static Vector<int> _pi32_256_4 = new Vector<int>(4);
        static Vector<int> _pi32_256_2 = new Vector<int>(2);
        static Vector<int> _pi32_256_1 = Vector<int>.One;
        static Vector<int> _pi32_256_0 = Vector<int>.Zero;

        /* the smallest non denormalized float number */
        static Vector<int> _ps256_min_norm_pos = new Vector<int>(0x00800000);
        static Vector<int> _ps256_inv_mant_mask = new Vector<int>(~0x7f800000);

        static Vector<float> _ps256_cephes_SQRTHF = new Vector<float>(0.707106781186547524f);
        static Vector<float> _ps256_cephes_log_p0 = new Vector<float>(7.0376836292E-2f);
        static Vector<float> _ps256_cephes_log_p1 = new Vector<float>(-1.1514610310E-1f);
        static Vector<float> _ps256_cephes_log_p2 = new Vector<float>(1.1676998740E-1f);
        static Vector<float> _ps256_cephes_log_p3 = new Vector<float>(-1.2420140846E-1f);
        static Vector<float> _ps256_cephes_log_p4 = new Vector<float>(+1.4249322787E-1f);
        static Vector<float> _ps256_cephes_log_p5 = new Vector<float>(-1.6668057665E-1f);
        static Vector<float> _ps256_cephes_log_p6 = new Vector<float>(+2.0000714765E-1f);
        static Vector<float> _ps256_cephes_log_p7 = new Vector<float>(-2.4999993993E-1f);
        static Vector<float> _ps256_cephes_log_p8 = new Vector<float>(+3.3333331174E-1f);
        static Vector<float> _ps256_cephes_log_q1 = new Vector<float>(-2.12194440e-4f);
        static Vector<float> _ps256_cephes_log_q2 = new Vector<float>(0.693359375f);

        public static Vector<float> Log(Vector<float> x)
        {
            var invalid_mask = Vector.LessThanOrEqual(x, zero);

            x = Vector.Max(x, Vector.AsVectorSingle(_ps256_min_norm_pos));  /* cut off denormalized stuff */

            var imm0 = AVXIntrinsics._mm256_srli_epi32(Vector.AsVectorInt32(x), 23);

            /* keep only the fractional part */
            x = Vector.BitwiseAnd(x, Vector.AsVectorSingle(_ps256_inv_mant_mask));
            x = Vector.BitwiseOr(x, _ps256_0p5);

            imm0 = Vector.Subtract(imm0, _pi32_256_0x7f);
            var e = AVXIntrinsics._mm256_cvtepi32_ps(imm0);

            e = Vector.Add(e, one);

            /* part2: 
               if( x < SQRTHF ) {
                 e -= 1;
                 x = x + x - 1.0;
               } else { x = x - 1.0; }
            */
            var mask = Vector.LessThan(x, _ps256_cephes_SQRTHF);
            var tmp = Vector.ConditionalSelect(mask, x, zero);
            x = Vector.Subtract(x, one);
            e = Vector.Subtract(e, Vector.ConditionalSelect(mask, one, zero));
            x = Vector.Add(x, tmp);

            var z = Vector.Multiply(x, x);

            var y = _ps256_cephes_log_p0;
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p1);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p2);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p3);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p4);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p5);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p6);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p7);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_log_p8);
            y = Vector.Multiply(y, x);

            y = Vector.Multiply(y, z);

            tmp = Vector.Multiply(e, _ps256_cephes_log_q1);
            y = Vector.Add(y, tmp);
            
            tmp = Vector.Multiply(z, _ps256_0p5);
            y = Vector.Subtract(y, tmp);

            tmp = Vector.Multiply(e, _ps256_cephes_log_q2);
            x = Vector.Add(x, y);
            x = Vector.Add(x, tmp);
            x = Vector.ConditionalSelect(invalid_mask, nan, x); // negative arg will be NAN
            return x;
        }

        static Vector<uint> _ps256_sign_mask = new Vector<uint>(0x80000000);
        static Vector<uint> _ps256_inv_sign_mask = new Vector<uint>(~0x80000000);

        static Vector<float> _ps256_minus_cephes_DP1 = new Vector<float>(-0.78515625f);
        static Vector<float> _ps256_minus_cephes_DP2 = new Vector<float>(-2.4187564849853515625e-4f);
        static Vector<float> _ps256_minus_cephes_DP3 = new Vector<float>(-3.77489497744594108e-8f);
        static Vector<float> _ps256_sincof_p0        = new Vector<float>(-1.9515295891E-4f);
        static Vector<float> _ps256_sincof_p1        = new Vector<float>( 8.3321608736E-3f);
        static Vector<float> _ps256_sincof_p2        = new Vector<float>(-1.6666654611E-1f);
        static Vector<float> _ps256_coscof_p0        = new Vector<float>( 2.443315711809948E-005f);
        static Vector<float> _ps256_coscof_p1        = new Vector<float>(-1.388731625493765E-003f);
        static Vector<float> _ps256_coscof_p2        = new Vector<float>( 4.166664568298827E-002f);
        static Vector<float> _ps256_cephes_FOPI      = new Vector<float>(1.27323954473516f); // 4 / M_PI

        public static Vector<float> Sin(Vector<float> x)
        {
            var sign_bit = x;
            /* take the absolute value */
            x = Vector.BitwiseAnd(x, Vector.AsVectorSingle(_ps256_inv_sign_mask));
            /* extract the sign bit (upper one) */
            sign_bit = Vector.BitwiseAnd(sign_bit, Vector.AsVectorSingle(_ps256_sign_mask));

            /* scale by 4/Pi */
            var y = Vector.Multiply(x, _ps256_cephes_FOPI);

            /* store the integer part of y in mm0 */
            var imm2 = AVXIntrinsics._mm256_cvttps_epi32(y);
            /* j=(j+1) & (~1) (see the cephes sources) */
            // another two AVX2 instruction
            imm2 = Vector.Add(imm2, _pi32_256_1);
            imm2 = Vector.BitwiseAnd(imm2, _pi32_256_inv1);
            y = AVXIntrinsics._mm256_cvtepi32_ps(imm2);

            /* get the swap sign flag */
            var imm0 = Vector.BitwiseAnd(imm2, _pi32_256_4);
            imm0 = AVXIntrinsics._mm256_slli_epi32(imm0, 29);
            /* get the polynom selection mask 
               there is one polynom for 0 <= x <= Pi/4
               and another one for Pi/4<x<=Pi/2

               Both branches will be computed.
            */
            imm2 = Vector.BitwiseAnd(imm2, _pi32_256_2);
            imm2 = Vector.Equals(imm2, _pi32_256_0);

            //var swap_sign_bit = Vector.AsVectorSingle(imm0);
            //var poly_mask = Vector.AsVectorSingle(imm2);
            sign_bit = Vector.AsVectorSingle(AVXIntrinsics._mm256_xor_si256(Vector.AsVectorInt32(sign_bit), imm0)); // swap_sign_bit);

            /* The magic pass: "Extended precision modular arithmetic" 
               x = ((x - y * DP1) - y * DP2) - y * DP3; */
            var xmm1 = _ps256_minus_cephes_DP1;
            var xmm2 = _ps256_minus_cephes_DP2;
            var xmm3 = _ps256_minus_cephes_DP3;
            xmm1 = Vector.Multiply(y, xmm1);
            xmm2 = Vector.Multiply(y, xmm2);
            xmm3 = Vector.Multiply(y, xmm3);
            x = Vector.Add(x, xmm1);
            x = Vector.Add(x, xmm2);
            x = Vector.Add(x, xmm3);

            /* Evaluate the first polynom  (0 <= x <= Pi/4) */
            y = _ps256_coscof_p0;
            var z = Vector.Multiply(x, x);

            y = Vector.Multiply(y, z);
            y = Vector.Add(y, _ps256_coscof_p1);
            y = Vector.Multiply(y, z);
            y = Vector.Add(y, _ps256_coscof_p2);
            y = Vector.Multiply(y, z);
            y = Vector.Multiply(y, z);
            var tmp = Vector.Multiply(z, _ps256_0p5);
            y = Vector.Subtract(y, tmp);
            y = Vector.Add(y, one);

            /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

            var y2 = _ps256_sincof_p0;
            y2 = Vector.Multiply(y2, z);
            y2 = Vector.Add(y2, _ps256_sincof_p1);
            y2 = Vector.Multiply(y2, z);
            y2 = Vector.Add(y2, _ps256_sincof_p2);
            y2 = Vector.Multiply(y2, z);
            y2 = Vector.Multiply(y2, x);
            y2 = Vector.Add(y2, x);

            /* select the correct result from the two polynoms */
            y = Vector.ConditionalSelect(imm2, y2, y);

            //y2 = Vector.BitwiseAnd(poly_mask, y2); //, xmm3);
            //y = _mm256_andnot_ps(poly_mask, y);
            //y = Vector.Add(y, y2);

            /* update the sign */
            y = Vector.AsVectorSingle(AVXIntrinsics._mm256_xor_si256(Vector.AsVectorInt32(y), Vector.AsVectorInt32(sign_bit)));

            return y;
        }

        public static Vector<float> Cos(Vector<float> x)
        {
            /* take the absolute value */
            x = Vector.BitwiseAnd(x, Vector.AsVectorSingle(_ps256_inv_sign_mask));

            /* scale by 4/Pi */
            var y = Vector.Multiply(x, _ps256_cephes_FOPI);

            /* store the integer part of y in mm0 */
            var imm2 = AVXIntrinsics._mm256_cvttps_epi32(y);
            /* j=(j+1) & (~1) (see the cephes sources) */
            imm2 = Vector.Add(imm2, _pi32_256_1);
            imm2 = Vector.BitwiseAnd(imm2, _pi32_256_inv1);
            y = AVXIntrinsics._mm256_cvtepi32_ps(imm2);
            imm2 = Vector.Subtract(imm2, _pi32_256_2);

            /* get the swap sign flag */
            var imm0 = AVXIntrinsics._mm256_andnot_si256(imm2, _pi32_256_4);
            imm0 = AVXIntrinsics._mm256_slli_epi32(imm0, 29);
            /* get the polynom selection mask 
               there is one polynom for 0 <= x <= Pi/4
               and another one for Pi/4<x<=Pi/2

               Both branches will be computed.
            */
            imm2 = Vector.BitwiseAnd(imm2, _pi32_256_2);
            imm2 = Vector.Equals(imm2, _pi32_256_0);

            //var sign_bit = Vector.AsVectorSingle(imm0);
            //var poly_mask = Vector.AsVectorSingle(imm2);

            /* The magic pass: "Extended precision modular arithmetic" 
               x = ((x - y * DP1) - y * DP2) - y * DP3; */
            var xmm1 = _ps256_minus_cephes_DP1;
            var xmm2 = _ps256_minus_cephes_DP2;
            var xmm3 = _ps256_minus_cephes_DP3;
            xmm1 = Vector.Multiply(y, xmm1);
            xmm2 = Vector.Multiply(y, xmm2);
            xmm3 = Vector.Multiply(y, xmm3);
            x = Vector.Add(x, xmm1);
            x = Vector.Add(x, xmm2);
            x = Vector.Add(x, xmm3);

            /* Evaluate the first polynom  (0 <= x <= Pi/4) */
            y = _ps256_coscof_p0;
            var z = Vector.Multiply(x, x);

            y = Vector.Multiply(y, z);
            y = Vector.Add(y, _ps256_coscof_p1);
            y = Vector.Multiply(y, z);
            y = Vector.Add(y, _ps256_coscof_p2);
            y = Vector.Multiply(y, z);
            y = Vector.Multiply(y, z);
            var tmp = Vector.Multiply(z, _ps256_0p5);
            y = Vector.Subtract(y, tmp);
            y = Vector.Add(y, one);

            /* Evaluate the second polynom  (Pi/4 <= x <= 0) */

            var y2 = _ps256_sincof_p0;
            y2 = Vector.Multiply(y2, z);
            y2 = Vector.Add(y2, _ps256_sincof_p1);
            y2 = Vector.Multiply(y2, z);
            y2 = Vector.Add(y2, _ps256_sincof_p2);
            y2 = Vector.Multiply(y2, z);
            y2 = Vector.Multiply(y2, x);
            y2 = Vector.Add(y2, x);

            /* select the correct result from the two polynoms */
            y = Vector.ConditionalSelect(imm2, y2, y);

            /* update the sign */
            y = Vector.AsVectorSingle(AVXIntrinsics._mm256_xor_si256(Vector.AsVectorInt32(y), imm0)); // Vector.AsVectorInt32(sign_bit)

            return y;
        }

        static Vector<float> _ps256_exp_hi = new Vector<float>(88.3762626647949f);
        static Vector<float> _ps256_exp_lo = new Vector<float>(-88.3762626647949f);

        static Vector<float> _ps256_cephes_LOG2EF = new Vector<float>(1.44269504088896341f);

        static Vector<float> _ps256_cephes_exp_C1 = new Vector<float>(0.693359375f);
        static Vector<float> _ps256_cephes_exp_C2 = new Vector<float>(-2.12194440e-4f);

        static Vector<float> _ps256_cephes_exp_p0 = new Vector<float>(1.9875691500E-4f);
        static Vector<float> _ps256_cephes_exp_p1 = new Vector<float>(1.3981999507E-3f);
        static Vector<float> _ps256_cephes_exp_p2 = new Vector<float>(8.3334519073E-3f);
        static Vector<float> _ps256_cephes_exp_p3 = new Vector<float>(4.1665795894E-2f);
        static Vector<float> _ps256_cephes_exp_p4 = new Vector<float>(1.6666665459E-1f);
        static Vector<float> _ps256_cephes_exp_p5 = new Vector<float>(5.0000001201E-1f);

        public static Vector<float> Exp(Vector<float> x)
        {
            x = Vector.Min(x, _ps256_exp_hi);
            x = Vector.Max(x, _ps256_exp_lo);

            /* express exp(x) as exp(g + n*log(2)) */
            var fx = Vector.Multiply(x, _ps256_cephes_LOG2EF);
            fx = fx + _ps256_0p5;

            var tmp = AVXIntrinsics._mm256_floor_ps(fx);

            /* if greater, substract 1 */
            var mask = Vector.GreaterThan(tmp, fx);  // ORIENTATION??
            var zeroOrOne = Vector.ConditionalSelect(mask, one, zero); // ORIENTATION??
            fx = Vector.Subtract(tmp, zeroOrOne);

            tmp = Vector.Multiply(fx, _ps256_cephes_exp_C1);
            var z = Vector.Multiply(fx, _ps256_cephes_exp_C2);
            x = Vector.Subtract(x, tmp);
            x = Vector.Subtract(x, z);

            z = Vector.Multiply(x, x);

            var y = _ps256_cephes_exp_p0;
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_exp_p1);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_exp_p2);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_exp_p3);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_exp_p4);
            y = Vector.Multiply(y, x);
            y = Vector.Add(y, _ps256_cephes_exp_p5);
            y = Vector.Multiply(y, z);
            y = Vector.Add(y, x);
            y = Vector.Add(y, one);

            /* build 2^n */
            var imm0 = AVXIntrinsics._mm256_cvttps_epi32(fx);

            // another two AVX2 instructions
            imm0 = Vector.Add(imm0, _pi32_256_0x7f);
            imm0 = AVXIntrinsics._mm256_slli_epi32(imm0, 23);
            var pow2n = Vector.AsVectorSingle(imm0);
            y = Vector.Multiply(y, pow2n);

            return y;
        }

    }

    public class Program
    {


        public static void Main(string[] args)
        {
            var len = Vector<float>.Count;

            var xss = Enumerable.Range(-100, 200)
                               .Select(i => i / 10.0f)
                               .ToArray();
            var offset = 0;
            while (offset + len <= xss.Length)
            {
                var xs = new Vector<float>(xss, offset);
                var ys = VectorMathFun.Exp(xs);
                for (int i = 0; i < len; i++)
                {
                    Console.WriteLine("Exp {0} {1} {2} {3}", xs[i], ys[i], Math.Exp(xs[i]), ys[i] - Math.Exp(xs[i]));
                }
                offset += len;
            }
            Console.WriteLine();

            offset = 0;
            while (offset + len <= xss.Length)
            {
                var xs = new Vector<float>(xss, offset);
                var ys = VectorMathFun.Log(xs);
                for (int i = 0; i < len; i++)
                {
                    Console.WriteLine("Log {0} {1} {2} {3}", xs[i], ys[i], Math.Log(xs[i]), ys[i] - Math.Log(xs[i]));
                }
                offset += len;
            }
            Console.WriteLine();

            offset = 0;
            while (offset + len <= xss.Length)
            {
                var xs = new Vector<float>(xss, offset);
                var ys = VectorMathFun.Sin(xs);
                for (int i = 0; i < len; i++)
                {
                    Console.WriteLine("Sin {0} {1} {2} {3}", xs[i], ys[i], Math.Sin(xs[i]), ys[i] - Math.Sin(xs[i]));
                }
                offset += len;
            }
            Console.WriteLine();

            offset = 0;
            while (offset + len <= xss.Length)
            {
                var xs = new Vector<float>(xss, offset);
                var ys = VectorMathFun.Cos(xs);
                for (int i = 0; i < len; i++)
                {
                    Console.WriteLine("Cos {0} {1} {2} {3}", xs[i], ys[i], Math.Cos(xs[i]), ys[i] - Math.Cos(xs[i]));
                }
                offset += len;
            }
            Console.WriteLine();
        }
    }
}
