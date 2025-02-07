#ifndef OPENCV_HAL_RVV_SQRT_HPP_INCLUDED
#define OPENCV_HAL_RVV_SQRT_HPP_INCLUDED
#include <riscv_vector.h>

using namespace std;
namespace cv { namespace cv_hal_rvv {


#undef cv_hal_sqrt32f
#define cv_hal_sqrt32f cv::cv_hal_rvv::sqrt32f
    
    inline int sqrt32f (const float *src, float *dst, int len) {
    const size_t vl = __riscv_vsetvl_e32m8(len);
    const size_t remainings = len % vl;
    auto calc_fun = [&](const size_t i, const size_t vl) {
        vfloat32m8_t vres;
        {
            vfloat32m8_t vsrc = __riscv_vle32_v_f32m8(&src[i], vl);
            vres = __riscv_vfsqrt_v_f32m8(vsrc, vl);
        }
        __riscv_vse32_v_f32m8(&dst[i], vres, vl);
    };
    size_t i = 0;
    for (; i < len - remainings; i += vl)
        calc_fun(i, vl);

    if (remainings){
        size_t tail_len = __riscv_vsetvl_e32m8(len - i);
        calc_fun(i, tail_len);
    }
    return CV_HAL_ERROR_OK;
}
}}
#endif