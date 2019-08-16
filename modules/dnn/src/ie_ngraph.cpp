// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "ie_ngraph.hpp"

#include <ie_ir_reader.hpp>

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_INF_ENGINE
#include <ie_extension.h>
#include <ie_plugin_dispatcher.hpp>
#endif  // HAVE_INF_ENGINE

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_INF_ENGINE

// For networks with input layer which has an empty name, IE generates a name id[some_number].
// OpenCV lets users use an empty input name and to prevent unexpected naming,
// we can use some predefined name.
static std::string kDefaultInpLayerName = "empty_inp_layer_name";

static std::vector<Ptr<NgraphBackendWrapper> >
ngraphWrappers(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<Ptr<NgraphBackendWrapper> > wrappers(ptrs.size());
    for (int i = 0; i < ptrs.size(); ++i)
    {
        CV_Assert(!ptrs[i].empty());
        wrappers[i] = ptrs[i].dynamicCast<NgraphBackendWrapper>();
        CV_Assert(!wrappers[i].empty());
    }
    return wrappers;
}

InfEngineNgraphNode::InfEngineNgraphNode(const std::shared_ptr<ngraph::Node>& _node)
    : BackendNode(DNN_BACKEND_NGRAPH), node(_node) {}

void InfEngineNgraphNode::setName(const std::string& name) {
    node->set_friendly_name(name);
}


InfEngineNgraphNet::InfEngineNgraphNet()
{
    hasNetOwner = false;
    device_name = "CPU";
}

InfEngineNgraphNet::InfEngineNgraphNet(InferenceEngine::CNNNetwork& net) : cnn(net)
{
    hasNetOwner = true;
    device_name = "CPU";
}

void InfEngineNgraphNet::init(int targetId)
{
    if (!hasNetOwner)
    {
        CV_Assert(!unconnectedNodes.empty());
        ngraph::ResultVector outs;
        for (auto node : unconnectedNodes)
        {
            auto out = std::make_shared<ngraph::op::Result>(node);
            outs.emplace_back(out);
        }
        std::cout << "outs  " << outs[0]->get_shape() << '\n';
        std::cout << "input " << inputs_vec[0]->get_shape() << '\n';
        ngraph_function = std::make_shared<ngraph::Function>(outs, inputs_vec);

        auto nodes = ngraph_function->get_ops(false);
        for (auto node : nodes) {
            std::cout << node->description() << '\n';
        }

        cnn = InferenceEngine::CNNNetwork(InferenceEngine::convertFunctionToICNNNetwork(ngraph_function));
    }

    switch (targetId)
    {
        case DNN_TARGET_CPU:
            device_name = "CPU";
            break;
        case DNN_TARGET_OPENCL:
        case DNN_TARGET_OPENCL_FP16:
            device_name = "GPU";
            break;
        case DNN_TARGET_MYRIAD:
            device_name = "MYRIAD";
            break;
        case DNN_TARGET_FPGA:
            device_name = "FPGA";
            break;
        default:
            CV_Error(Error::StsNotImplemented, "Unknown target");
    };
    for (const auto& name : requestedOutputs)
    {
        cnn.addOutput(name);
    }
    initPlugin(cnn);
}

void InfEngineNgraphNet::setInputs(const std::vector<cv::Mat>& inputs, const std::vector<std::string>& names) {
    CV_Assert_N(!inputs.empty(), inputs.size() == names.size());
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<size_t> shape(&inputs[i].size[0], &inputs[i].size[0] + inputs[i].dims);
        auto type = inputs[i].type() == CV_16S ?
                    ngraph::element::f16 : ngraph::element::f32;
        auto inp = std::make_shared<ngraph::op::Parameter>(type, ngraph::Shape(shape));
        inp->set_friendly_name(names[i]);
        inputs_vec.push_back(inp);
    }
}

ngraph::ParameterVector InfEngineNgraphNet::getInputs() {
    return inputs_vec;
}

void InfEngineNgraphNet::setUnconnectedNodes(Ptr<InfEngineNgraphNode> node) {
    unconnectedNodes.insert(node->node);
}

static InferenceEngine::Core& getCore()
{
    static InferenceEngine::Core core;
    return core;
}

#if !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
static bool detectMyriadX_()
{
    auto input = std::make_shared<ngraph::op::Parameter>(ngraph::element::f16, ngraph::Shape({1}));
    auto relu = std::make_shared<ngraph::op::Relu>(input);
    auto ngraph_function = std::make_shared<ngraph::Function>(relu, ngraph::ParameterVector{input});
    InferenceEngine::CNNNetwork cnn = InferenceEngine::CNNNetwork(
                                      InferenceEngine::convertFunctionToICNNNetwork(ngraph_function));
    try
    {
        auto netExec = getCore().LoadNetwork(cnn, "MYRIAD", {{"VPU_PLATFORM", "VPU_2480"}});
        auto infRequest = netExec.CreateInferRequest();
    } catch(...) {
        return false;
    }
    return true;
}
#endif  // !defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)


#ifdef HAVE_INF_ENGINE
bool isMyriadX()
{
     static bool myriadX = getInferenceEngineVPUType() == CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;
     return myriadX;
}

static std::string getInferenceEngineVPUType_()
{
    static std::string param_vpu_type = utils::getConfigurationParameterString("OPENCV_DNN_IE_VPU_TYPE", "");
    if (param_vpu_type == "")
    {
#if defined(OPENCV_DNN_IE_VPU_TYPE_DEFAULT)
        param_vpu_type = OPENCV_DNN_IE_VPU_TYPE_DEFAULT;
#else
        CV_LOG_INFO(NULL, "OpenCV-DNN: running Inference Engine VPU autodetection: Myriad2/X. In case of other accelerator types specify 'OPENCV_DNN_IE_VPU_TYPE' parameter");
        try {
            bool isMyriadX_ = detectMyriadX_();
            if (isMyriadX_)
            {
                param_vpu_type = CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_X;
            }
            else
            {
                param_vpu_type = CV_DNN_INFERENCE_ENGINE_VPU_TYPE_MYRIAD_2;
            }
        }
        catch (...)
        {
            CV_LOG_WARNING(NULL, "OpenCV-DNN: Failed Inference Engine VPU autodetection. Specify 'OPENCV_DNN_IE_VPU_TYPE' parameter.");
            param_vpu_type.clear();
        }
#endif
    }
    CV_LOG_INFO(NULL, "OpenCV-DNN: Inference Engine VPU type='" << param_vpu_type << "'");
    return param_vpu_type;
}

cv::String getInferenceEngineVPUType()
{
    static cv::String vpu_type = getInferenceEngineVPUType_();
    return vpu_type;
}
#else  // HAVE_INF_ENGINE
cv::String getInferenceEngineVPUType()
{
    CV_Error(Error::StsNotImplemented, "This OpenCV build doesn't include InferenceEngine support");
}
#endif  // HAVE_INF_ENGINE

void resetMyriadDevice()
{
#ifdef HAVE_INF_ENGINE
    AutoLock lock(getInitializationMutex());
    getCore().UnregisterPlugin("MYRIAD");
#endif  // HAVE_INF_ENGINE
}

void InfEngineNgraphNet::initPlugin(InferenceEngine::CNNNetwork& net)
{
    CV_Assert(!isInitialized());

    try
    {
        AutoLock lock(getInitializationMutex());
        {
            isInit = true;

            std::vector<std::string> candidates;
            std::string param_pluginPath = utils::getConfigurationParameterString("OPENCV_DNN_IE_EXTRA_PLUGIN_PATH", "");
            if (!param_pluginPath.empty())
            {
                candidates.push_back(param_pluginPath);
            }

            if (device_name == "CPU" || device_name == "FPGA")
            {
                std::string suffixes[] = {"_avx2", "_sse4", ""};
                bool haveFeature[] = {
                    checkHardwareSupport(CPU_AVX2),
                    checkHardwareSupport(CPU_SSE4_2),
                    true
                };
                for (int i = 0; i < 3; ++i)
                {
                    if (!haveFeature[i])
                        continue;
#ifdef _WIN32
                    candidates.push_back("cpu_extension" + suffixes[i] + ".dll");
#elif defined(__APPLE__)
                    candidates.push_back("libcpu_extension" + suffixes[i] + ".so");  // built as loadable module
                    candidates.push_back("libcpu_extension" + suffixes[i] + ".dylib");  // built as shared library
#else
                    candidates.push_back("libcpu_extension" + suffixes[i] + ".so");
#endif  // _WIN32
                }
            }
            bool found = false;
            for (size_t i = 0; i != candidates.size(); ++i)
            {
                const std::string& libName = candidates[i];
                try
                {
                    InferenceEngine::IExtensionPtr extension =
                        InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(libName);

                    getCore().AddExtension(extension, "CPU");

                    CV_LOG_INFO(NULL, "DNN-IE: Loaded extension plugin: " << libName);
                    found = true;
                    break;
                }
                catch(...) {}
            }
            if (!found && !candidates.empty())
            {
                CV_LOG_WARNING(NULL, "DNN-IE: Can't load extension plugin (extra layers for some networks). Specify path via OPENCV_DNN_IE_EXTRA_PLUGIN_PATH parameter");
            }
            // Some of networks can work without a library of extra layers.
#ifndef _WIN32
            // Limit the number of CPU threads.
            if (device_name == "CPU")
                getCore().SetConfig({{
                    InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, format("%d", getNumThreads()),
                }}, device_name);
#endif
        }
        netExec = getCore().LoadNetwork(net, device_name);
    }
    catch (const std::exception& ex)
    {
        CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
    }
}

bool InfEngineNgraphNet::isInitialized()
{
    return isInit;
}

bool NgraphBackendLayer::getMemoryShapes(const std::vector<MatShape> &inputs,
                                            const int requiredOutputs,
                                            std::vector<MatShape> &outputs,
                                            std::vector<MatShape> &internals) const
{
    InferenceEngine::ICNNNetwork::InputShapes inShapes = t_net.getInputShapes();
    InferenceEngine::ICNNNetwork::InputShapes::iterator itr;
    bool equal_flag = true;
    size_t i = 0;
    for (itr = inShapes.begin(); itr != inShapes.end(); ++itr)
    {
        InferenceEngine::SizeVector currentInShape(inputs[i].begin(), inputs[i].end());
        if (itr->second != currentInShape)
        {
            itr->second = currentInShape;
            equal_flag = false;
        }
        i++;
    }

    if (!equal_flag)
    {
        InferenceEngine::CNNNetwork curr_t_net(t_net);
        curr_t_net.reshape(inShapes);
    }
    std::vector<size_t> dims = t_net.getOutputsInfo()[name]->getDims();
    outputs.push_back(MatShape(dims.begin(), dims.end()));
    return false;
}

bool NgraphBackendLayer::supportBackend(int backendId)
{
    return backendId == DNN_BACKEND_DEFAULT ||
           ((backendId == DNN_BACKEND_INFERENCE_ENGINE || backendId == DNN_BACKEND_NGRAPH));
}

void NgraphBackendLayer::forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                                    OutputArrayOfArrays internals)
{
    CV_Error(Error::StsInternal, "Choose Inference Engine as a preferable backend.");
}


void addConstantData(const ngraph::element::Type &type, ngraph::Shape shape, const void *data) {
    auto node = std::make_shared<ngraph::op::Constant>(type, shape, data);
}

Mat convertFp16(const std::shared_ptr<ngraph::Node>& node)
{
    CV_Assert(node->is_constant());

    std::vector<float> dataFp32 = std::dynamic_pointer_cast<ngraph::op::Constant>(node)->get_vector<float>();
    Mat src(dataFp32);
    Mat dst(1, dataFp32.size(), CV_16SC1);
    convertFp16(src, dst);
    return dst;
}

static InferenceEngine::Layout estimateLayout(const Mat& m)
{
    if (m.dims == 4)
        return InferenceEngine::Layout::NCHW;
    else if (m.dims == 2)
        return InferenceEngine::Layout::NC;
    else
        return InferenceEngine::Layout::ANY;
}

static InferenceEngine::DataPtr wrapToInfEngineDataNode(const Mat& m, const std::string& name = "")
{
    std::vector<size_t> shape(&m.size[0], &m.size[0] + m.dims);
    if (m.type() == CV_32F)
        return InferenceEngine::DataPtr(new InferenceEngine::Data(name,
               {InferenceEngine::Precision::FP32, shape, estimateLayout(m)}));
    else if (m.type() == CV_8U)
        return InferenceEngine::DataPtr(new InferenceEngine::Data(name,
               {InferenceEngine::Precision::U8, shape, estimateLayout(m)}));
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %d", m.type()));
}

InferenceEngine::Blob::Ptr wrapToNgraphBlob(const Mat& m, const std::vector<size_t>& shape,
                                               InferenceEngine::Layout layout)
{
    if (m.type() == CV_32F)
        return InferenceEngine::make_shared_blob<float>(
               {InferenceEngine::Precision::FP32, shape, layout}, (float*)m.data);
    else if (m.type() == CV_8U)
        return InferenceEngine::make_shared_blob<uint8_t>(
               {InferenceEngine::Precision::U8, shape, layout}, (uint8_t*)m.data);
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %d", m.type()));
}

InferenceEngine::Blob::Ptr wrapToNgraphBlob(const Mat& m, InferenceEngine::Layout layout)
{
    std::vector<size_t> shape(&m.size[0], &m.size[0] + m.dims);
    return wrapToNgraphBlob(m, shape, layout);
}

NgraphBackendWrapper::NgraphBackendWrapper(int targetId, const cv::Mat& m)
    : BackendWrapper(DNN_BACKEND_NGRAPH, targetId)
{
    dataPtr = wrapToInfEngineDataNode(m);
    blob = wrapToNgraphBlob(m, estimateLayout(m));
}

NgraphBackendWrapper::NgraphBackendWrapper(Ptr<BackendWrapper> wrapper)
    : BackendWrapper(DNN_BACKEND_NGRAPH, wrapper->targetId)
{
    Ptr<NgraphBackendWrapper> ieWrapper = wrapper.dynamicCast<NgraphBackendWrapper>();
    CV_Assert(!ieWrapper.empty());
    InferenceEngine::DataPtr srcData = ieWrapper->dataPtr;
    dataPtr = InferenceEngine::DataPtr(new InferenceEngine::Data(srcData->getName(), srcData->getTensorDesc()));
    blob = ieWrapper->blob;
}

Ptr<BackendWrapper> NgraphBackendWrapper::create(Ptr<BackendWrapper> wrapper)
{
    return Ptr<BackendWrapper>(new NgraphBackendWrapper(wrapper));
}

NgraphBackendWrapper::~NgraphBackendWrapper()
{

}

void NgraphBackendWrapper::copyToHost()
{

}

void NgraphBackendWrapper::setHostDirty()
{

}

InferenceEngine::DataPtr ngraphDataNode(const Ptr<BackendWrapper>& ptr)
{
    CV_Assert(!ptr.empty());
    Ptr<NgraphBackendWrapper> p = ptr.dynamicCast<NgraphBackendWrapper>();
    CV_Assert(!p.empty());
    return p->dataPtr;
}


void forwardNgraph(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                      Ptr<BackendNode>& node)
{
#ifdef HAVE_INF_ENGINE
    CV_Assert(!node.empty());
    Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
    CV_Assert(!ieNode.empty());
    ieNode->net->forward(outBlobsWrappers);
#endif  // HAVE_INF_ENGINE
}

void InfEngineNgraphNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = ngraphWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        std::string name = wrapper->dataPtr->getName();
        name = name.empty() ? kDefaultInpLayerName : name;
        allBlobs.insert({name, wrapper->blob});
    }
}


void InfEngineNgraphNet::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrapper)
{
    // Look for finished requests.
    Ptr<NgraphReqWrapper> reqWrapper;
    for (auto& wrapper : infRequests)
    {
        if (wrapper->isReady)
        {
            reqWrapper = wrapper;
            break;
        }
    }
    if (reqWrapper.empty())
    {
        reqWrapper = Ptr<NgraphReqWrapper>(new NgraphReqWrapper());
        try
        {
            reqWrapper->req = netExec.CreateInferRequest();
        }
        catch (const std::exception& ex)
        {
            CV_Error(Error::StsAssert, format("Failed to initialize Inference Engine backend: %s", ex.what()));
        }
        infRequests.push_back(reqWrapper);

        InferenceEngine::BlobMap inpBlobs, outBlobs;
        for (const auto& it : cnn.getInputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            inpBlobs[name] = blobIt->second;
        }
        for (const auto& it : cnn.getOutputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            outBlobs[name] = blobIt->second;
        }
        reqWrapper->req.SetInput(inpBlobs);
        reqWrapper->req.SetOutput(outBlobs);

        // InferenceEngine::IInferRequest::Ptr infRequestPtr = reqWrapper->req;
        // infRequestPtr->SetUserData(reqWrapper.get(), 0);

    }

    reqWrapper->req.Infer();
}

#endif

}}
