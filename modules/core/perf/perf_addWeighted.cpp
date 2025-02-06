#include "perf_precomp.hpp"
#include <algorithm>

namespace opencv_test {
using namespace perf;
using namespace std;

#define TYPICAL_MAT_TYPES_ADWEIGHTED CV_8UC1
#define TYPICAL_MATS_ADWEIGHTED testing::Combine(testing::Values(szVGA, sz720p, sz1080p), testing::Values(TYPICAL_MAT_TYPES_ADWEIGHTED))

PERF_TEST_P(Size_MatType, addWeighted, TYPICAL_MATS_ADWEIGHTED)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());
    int depth = CV_MAT_DEPTH(type);
    Mat src1(size, type);
    Mat src2(size, type);
    double alpha = 3.75;
    double beta = -0.125;
    double gamma = 100.0;
    Mat dst(size, type);
    declare.in(src1, src2, dst, WARMUP_RNG).out(dst);

    if (depth == CV_32S)
    {
        src1 /= 2048;
        src2 /= 2048;
    }

    TEST_CYCLE() cv::addWeighted(src1, alpha, src2, beta, gamma, dst, dst.type());
    SANITY_CHECK(dst, depth == CV_32S ? 4 : 1);
}

// Тест для addWeighted8u
PERF_TEST_P(Size_MatType, AddWeighted_U8, TYPICAL_MATS_ADWEIGHTED)
{
    Size size = get<0>(GetParam());
    int type = get<1>(GetParam());

    if (type != CV_8UC1)
    {
        SANITY_CHECK_NOTHING();
        std::cout << "Skipping test for unsupported type: " << type << std::endl;
        return;
    }

    Mat src1(size, type);
    Mat src2(size, type);
    Mat dst(size, type);
    Mat ref(size, type);

    double alpha = 0.75f;
    double beta = 0.25f;
    double gamma = 50.0f;

    {
        SANITY_CHECK_NOTHING();
        std::cout << "Skipping test for unsupported type: " << type << std::endl;
        return;
    }

    declare.in(src1, src2, WARMUP_RNG).out(dst);
    float scalars[] = { static_cast<float>(alpha), static_cast<float>(beta), static_cast<float>(gamma) };

    TEST_CYCLE()
    {
        cv::addWeighted( src1, alpha, src2, beta, gamma, dst);
    }

    cv::addWeighted(src1, alpha, src2, beta, gamma, ref, dst.type());

    SANITY_CHECK_NOTHING();
}

} // namespace opencv_test