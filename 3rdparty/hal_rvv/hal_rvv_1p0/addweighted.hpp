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
        const uint8_t* p_src1 = reinterpret_cast<const uint8_t*>(src1 + (j / width) * step1 + (j % width));
        const uint8_t* p_src2 = reinterpret_cast<const uint8_t*>(src2 + (j / width) * step2 + (j % width));
        uint8_t* p_dst = reinterpret_cast<uint8_t*>(dst + (j / width) * step + (j % width));

        // Load data from src1 and src2
        vuint8m1_t v_row1 = __riscv_vle8_v_u8m1(p_src1, vl);
        vuint8m1_t v_row2 = __riscv_vle8_v_u8m1(p_src2, vl);

        // Convert to 16-bit signed integers
        vuint16m2_t v_row1_u16 = __riscv_vwcvtu_x_x_v_u16m2(v_row1, vl);
        vuint16m2_t v_row2_u16 = __riscv_vwcvtu_x_x_v_u16m2(v_row2, vl);
        
        vint16m2_t v_row1_i16 = __riscv_vreinterpret_v_u16m2_i16m2(v_row1_u16);
        vint16m2_t v_row2_i16 = __riscv_vreinterpret_v_u16m2_i16m2(v_row2_u16);

        // Extend to 32-bit signed integers
        vint32m4_t v_row1_i32 = __riscv_vwcvt_x_x_v_i32m4(v_row1_i16, vl);
        vint32m4_t v_row2_i32 = __riscv_vwcvt_x_x_v_i32m4(v_row2_i16, vl);

        // // Convert to 32-bit floating-point vectors
        // vfloat32m4_t v_alpha = __riscv_vfmv_v_f_f32m4(alpha, vl);
        // vfloat32m4_t v_beta = __riscv_vfmv_v_f_f32m4(beta, vl);
        // vfloat32m4_t v_gamma = __riscv_vfmv_v_f_f32m4(gamma, vl);

        vfloat32m4_t v_row1_f = __riscv_vfcvt_f_x_v_f32m4(v_row1_i32, vl);
        vfloat32m4_t v_row2_f = __riscv_vfcvt_f_x_v_f32m4(v_row2_i32, vl);

        // Apply coefficients alpha, beta, and gamma
        vfloat32m4_t v_res_f = __riscv_vfmul_vf_f32m4(v_row1_f, alpha, vl);
        v_res_f = __riscv_vfmacc_vf_f32m4(v_res_f, beta, v_row2_f, vl);
        v_res_f = __riscv_vfadd_vf_f32m4(v_res_f, gamma, vl);

        // Clamp results to [0, 255]
        v_res_f = __riscv_vfmax_vf_f32m4(v_res_f, 0.0f, vl);
        v_res_f = __riscv_vfmin_vf_f32m4(v_res_f, 255.0f, vl);

        // Convert vfloat32m4_t to vuint8m1_t
        vint32m4_t v_dst_i32 = __riscv_vfcvt_rtz_x_f_v_i32m4(v_res_f, vl); 
        vint16m2_t v_dst_i16 = __riscv_vncvt_x_x_w_i16m2(v_dst_i32, vl);
        vint8m1_t v_dst_i8 = __riscv_vncvt_x_x_w_i8m1(v_dst_i16, vl);
        vuint8m1_t v_dst_u8 = __riscv_vreinterpret_v_i8m1_u8m1(v_dst_i8);

        // Store result
        __riscv_vse8_v_u8m1(p_dst, v_dst_u8, vl);

        j += vl;
    }
    return 0;
}  
} // namespace cv_hal_rvv 
} // namespace cv 
 
#endif // OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED
