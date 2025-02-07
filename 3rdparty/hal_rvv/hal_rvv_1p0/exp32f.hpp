#ifndef OPENCV_HAL_RVV_EXP32F_HPP_INCLUDED
#define OPENCV_HAL_RVV_EXP32F_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_exp32f
#define cv_hal_exp32f cv::cv_hal_rvv::exp32f

inline void exp32f(const float* src, float* dst, int n) {
    ssize_t i = 0;
    ssize_t vlmax = __riscv_vsetvlmax_e32m1();

    while(i < n) {
            ssize_t vl = n - i;
            if(vl > vlmax) vl = vlmax;
            vl = __riscv_vsetvl_e32m1(vl);

            vfloat32m1_t v_x = __riscv_vle32_v_f32m1(&src[i], vl);

            vfloat32m1_t v_result = __riscv_vfmv_v_f_f32m1(1.0f, vl);
            vfloat32m1_t v_term = v_result;

            for(ssize_t step = 1; step < 100; ++step) {
                    v_term = __riscv_vfmul_vv_f32m1(v_term, v_x, vl);
                    vfloat32m1_t v_n = __riscv_vfmv_v_f_f32m1(static_cast<float>(step), vl);
                    v_term = __riscv_vfdiv_vv_f32m1(v_term, v_n, vl);
                    v_result = __riscv_vfadd_vv_f32m1(v_result, v_term, vl);
            }
            __riscv_vse32_v_f32m1(&result[i], v_result, vl);
            i += vl;
    }
    if(i < n) {
            ssize_t remaining = n - i;
            vfloat32m1_t v_x = __riscv_vle32_v_f32m1(&src[i], remaining);

            vfloat32m1_t v_result = __riscv_vfmv_v_f_f32m1(1.0f, remaining);
            vfloat32m1_t v_term = v_result;

            for(ssize_t step = 1; step < 100; ++step) {
                    v_term = __riscv_vfmul_vv_f32m1(v_term, v_x, remaining);
                    vfloat32m1_t v_n = __riscv_vfmv_v_f_f32m1(static_cast<float>(step),  remaining);
                    v_term = __riscv_vfdiv_vv_f32m1(v_term, v_n, remaining);
                    v_result = __riscv_vfadd_vv_f32m1(v_result, v_term, remaining);
            }
            __riscv_vse32_v_f32m1(&result[i], v_result, remaining);
    }

}

} }

#endif