#include <opencv2/dnn/dnn.hpp>

#ifndef OPENCV_DNN_DNN_ZOO_HPP
#define OPENCV_DNN_DNN_ZOO_HPP

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

namespace zoo {

class CV_EXPORTS_W_SIMPLE Topology
{
public:
    CV_WRAP Topology(const String& modelURL, const String& configURL);
};

CV_EXPORTS_W void mymethod(int val);

}  // namespace cv::dnn::zoo

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

#endif  // OPENCV_DNN_DNN_ZOO_HPP
