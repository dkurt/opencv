#ifndef OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED 
#define OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED 

#include <riscv_vector.h> 

namespace cv { namespace cv_hal_rvv { 

#undef cv_hal_addWeighted8u 
#define cv_hal_addWeighted8u cv::cv_hal_rvv::addWeighted8u 

inline int addWeighted8u(const uchar* src1, size_t step1, const uchar* src2, size_t step2, 
    uchar* dst, size_t step, int width, int height, const void* _scalars) {
    const float* scalars = static_cast<const float*>(_scalars); 
    int alpha = 1; 
    int beta = 1; 
    int gamma = 0;
    
    int total_elements = width * height;
    int j = 0;

    while (j < total_elements) {
        size_t vl = __riscv_vsetvl_e8m1(total_elements - j);

        // Calculate the 1D index for src1, src2, and dst 
        const uint8_t* p_src1 = src1 + j;
        const uint8_t* p_src2 = src2 + j;
        uint8_t* p_dst = dst + j;

        vuint8m1_t v_row1 = __riscv_vle8_v_u8m1(p_src1, vl);
        vuint8m1_t v_row2 = __riscv_vle8_v_u8m1(p_src2, vl);

        vuint16m2_t v_row1_w = __riscv_vwcvtu_x_x_v_u16m2(v_row1, vl);
        vuint16m2_t v_row2_w = __riscv_vwcvtu_x_x_v_u16m2(v_row2, vl);

        // Compute weighted sum
        vuint16m2_t v_res = __riscv_vmul_vx_u16m2(v_row1_w, alpha, vl);
        v_res = __riscv_vwmaccu_vx_u16m2(v_res, beta, v_row2, vl);
        v_res = __riscv_vadd_vx_u16m2(v_res, gamma, vl);

        // Clamp results to [0, 255]
        v_res = __riscv_vmaxu_vx_u16m2(v_res, 0.0f, vl);
        v_res = __riscv_vminu_vx_u16m2(v_res, 255.0f, vl);

        vuint8m1_t v_dst_u8 = __riscv_vncvt_x_x_w_u8m1(v_res, vl);

        __riscv_vse8_v_u8m1(p_dst, v_dst_u8, vl);

        j += vl;
    }
    return 0;
    }

} // namespace cv_hal_rvv 
} // namespace cv 
#endif // OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED
