#ifndef OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED
#define OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_addWeighted8u
#define cv_hal_addWeighted8u cv::cv_hal_rvv::addWeighted8u

inline int addWeighted8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2,
    uchar* dst, size_t step, int width, int height, const void* _scalars) {
    const float* scalars = static_cast<const float*>(_scalars);
    float alpha = scalars[0];
    float beta = scalars[1];
    float gamma = scalars[2];

    int total_elements = width * height; // Total number of elements in the matrix

    int j = 0;
    while (j < total_elements) {
        size_t vl = __riscv_vsetvl_e8m1(total_elements - j);

        // Calculate the 1D index for src1, src2, and dst
        const uint8_t* p_src1 = src1 + j;
        const uint8_t* p_src2 = src2 + j;
        uint8_t* p_dst = dst + j;

        // Load data from src1 and src2
        vuint8m1_t v_row1 = __riscv_vle8_v_u8m1(p_src1, vl);
        vuint8m1_t v_row2 = __riscv_vle8_v_u8m1(p_src2, vl);

        // Convert to 16-bit unsigned integers
        vuint16m2_t v_row1_w = __riscv_vwcvtu_x_x_v_u16m2(v_row1, vl);
        vuint16m2_t v_row2_w = __riscv_vwcvtu_x_x_v_u16m2(v_row2, vl);

        // Reinterpret as signed 16-bit integers
        vint16m2_t v_row1_ext = __riscv_vreinterpret_v_u16m2_i16m2(v_row1_w);
        vint16m2_t v_row2_ext = __riscv_vreinterpret_v_u16m2_i16m2(v_row2_w);

        // Convert to 32-bit floating-point
        vfloat32m2_t v_row1_f = __riscv_vfcvt_f_x_v_f32m2(v_row1_ext, vl);
        vfloat32m2_t v_row2_f = __riscv_vfcvt_f_x_v_f32m2(v_row2_ext, vl);

        // Apply alpha, beta, and gamma
        vfloat32m2_t v_res_f = __riscv_vfmul_vf_f32m2(v_row1_f, alpha, vl);
        v_res_f = __riscv_vfmacc_vf_f32m2(v_res_f, beta, v_row2_f, vl);
        v_res_f = __riscv_vfadd_vf_f32m2(v_res_f, gamma, vl);

        // Clamp the result to [0, 255]
        v_res_f = __riscv_vfmax_vf_f32m2(v_res_f, 0.0f, vl);
        v_res_f = __riscv_vfmin_vf_f32m2(v_res_f, 255.0f, vl);

        // Convert back to 32-bit signed integers
        vint32m2_t v_res_i32 = __riscv_vfcvt_x_f_v_i32m2(v_res_f, vl);

        // Convert to 16-bit signed integers
        vint16m1_t v_res_i16 = __riscv_vncvt_x_x_w_i16m1(v_res_i32, vl);

        // Convert to 8-bit unsigned integers
        vuint8m1_t v_dst = __riscv_vncvt_x_x_w_u8m1(v_res_i16, vl);

        // Store the result
        __riscv_vse8_v_u8m1(p_dst, v_dst, vl);

        j += vl;
    }
    return 0;
}

} // namespace cv_hal_rvv
} // namespace cv


#endif // OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED
