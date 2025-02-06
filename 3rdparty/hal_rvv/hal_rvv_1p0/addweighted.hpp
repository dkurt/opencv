#ifndef OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED
#define OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

#undef cv_hal_addWeighted8u
#define cv_hal_addWeighted8u cv::cv_hal_rvv::addWeighted8u

inline int addWeighted8u( const uchar* src1, size_t step1, const uchar* src2, size_t step2, 
                        uchar* dst, size_t step, int width, int height, const void* _scalars ) {
    const float* scalars = static_cast<const float*>(_scalars);
    float alpha = scalars[0];
    float beta = scalars[1];
    float gamma = scalars[2];

    for (int i = 0; i < height; ++i) {
        const uint8_t* row1 = src1 + i * step1;
        const uint8_t* row2 = src2 + i * step2;
        uint8_t* row_dst = dst + i * step;
    
        int j = 0;
        while (j < width) {
            size_t vl = __riscv_vsetvl_e8m1(width - j);

            vuint8m1_t v_row1 = __riscv_vle8_v_u8m1(row1 + j, vl);
            vuint8m1_t v_row2 = __riscv_vle8_v_u8m1(row2 + j, vl);

            vuint16m2_t v_row1_w = __riscv_vwcvtu_x_x_v_u16m2(v_row1, vl);
            vuint16m2_t v_row2_w = __riscv_vwcvtu_x_x_v_u16m2(v_row2, vl);

            vint16m2_t v_row1_ext = __riscv_vreinterpret_v_u16m2_i16m2(v_row1_w);
            vint16m2_t v_row2_ext = __riscv_vreinterpret_v_u16m2_i16m2(v_row2_w);

            // Применение коэффициентов alpha, beta и gamma
            vint16m2_t v_res = __riscv_vmul_vx_i16m2(v_row1_ext, alpha, vl);
            v_res = __riscv_vmacc_vx_i16m2(v_res, beta, v_row2_ext, vl);
            v_res = __riscv_vadd_vx_i16m2(v_res, gamma, vl);
            
            // Ограничение результатов в пределах [0, 255]
            v_res = __riscv_vmax_vx_i16m2(v_res, 0, vl);
            v_res = __riscv_vmin_vx_i16m2(v_res, 255, vl);

            // Преобразование обратно в 16-беззнаковый формат
            vuint16m2_t v_res_unsigned = __riscv_vreinterpret_v_i16m2_u16m2(v_res);

            vuint8m1_t shift_vec = __riscv_vmv_v_x_u8m1(8, vl);
            vuint8m1_t v_dst = __riscv_vnclipu_wv_u8m1(v_res_unsigned, shift_vec, 0, vl);

            __riscv_vse8_v_u8m1(row_dst + j, v_dst, vl);

            j += vl;
        }
    }
}

} // namespace cv_hal_rvv
} // namespace cv

#endif // OPENCV_HAL_RVV_ADDWEIGHTED_HPP_INCLUDED