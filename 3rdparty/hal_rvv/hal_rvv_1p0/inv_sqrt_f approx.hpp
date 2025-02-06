// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#ifndef OPENCV_HAL_RVV_INV_SQRT_HPP_INCLUDED
#define OPENCV_HAL_RVV_INV_SQRT_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_invSqrt32f
#define cv_hal_invSqrt32f cv::cv_hal_rvv::invSqrt32f
// #undef cv_hal_invSqrt64f
// #define cv_hal_invSqrt64f cv::cv_hal_rvv::invSqrt64f

inline int invSqrt32f (const float *src, float *dst, int len) {
    const size_t vl = __riscv_vsetvl_e32m4(len);
    const size_t remainings = len % vl;
    auto calc_fun = [&](const size_t i, const size_t vl) {
        vfloat32m4_t vres;
        {
            const vfloat32m4_t vsrc = __riscv_vle32_v_f32m4(&src[i], vl);
            const vfloat64m8_t v_conv_64f = __riscv_vfwcvt_f_f_v_f64m8(vsrc, vl);
            vres = __riscv_vfncvt_f_f_w_f32m4(__riscv_vfrsqrt7_v_f64m8(v_conv_64f, vl), vl);
        }

        // Newton's: x_n+1 = x_n*(1.5 − 0.5*a*x_n^2)
        /*auto find_a = [vl](vfloat32m8_t vres) {
            const auto x_prev = vres;
            vres = __riscv_vfmul_vv_f32m8(vres, x_prev, vl);
            vres = __riscv_vfrdiv_vf_f32m8(vres, -0.5f * 1.25f, vl);
            return vres;
        };
        const auto a = find_a(vres);
        for (auto k = 0; k < 10000; ++k) {
            const auto x_prev = vres;
            vres = __riscv_vfmul_vv_f32m8( vres, x_prev, vl);
            vres = __riscv_vfmul_vv_f32m8( vres, a, vl);
            // vres = __riscv_vfmsac(-1.5f, -0.5f, vres, vl);
            vres = __riscv_vfadd_vf_f32m8( vres, 1.5f, vl);
            vres = __riscv_vfmul_vv_f32m8( vres, x_prev, vl);
        }*/
        __riscv_vse32_v_f32m4(&dst[i], vres, vl);
    };

    size_t i = 0;
    for (; i < len - remainings; i += vl)
        calc_fun(i, vl);
    if (remainings){
        size_t tail_len = __riscv_vsetvl_e32m4(len - i);
        calc_fun(i, tail_len);
    }
    return CV_HAL_ERROR_OK;
}

}}
#endif