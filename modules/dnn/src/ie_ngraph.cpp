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

InfEngineNgraphNode::InfEngineNgraphNode(std::shared_ptr<ngraph::Node>&& _node)
    : BackendNode(DNN_BACKEND_NGRAPH), node(std::move(_node)) {}

    InfEngineNgraphNode::InfEngineNgraphNode(std::shared_ptr<ngraph::Node>& _node)
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

void InfEngineNgraphNet::addOutput(const std::string& name)
{
    requestedOutputs.push_back(name);
}

void InfEngineNgraphNet::setNodePtr(std::shared_ptr<ngraph::Node>* ptr) {
    all_nodes.emplace((*ptr)->get_friendly_name(), ptr);
}

 void InfEngineNgraphNet::release() {
     for (auto& node : components.back()) {
         if (!(node->is_parameter() || node->is_output() || node->is_constant()) ) {
             auto it = all_nodes.find(node->get_friendly_name());
             if (it != all_nodes.end()) {
                 unconnectedNodes.erase(*(it->second));
                 it->second->reset();
                 all_nodes.erase(it);
             }
         }
     }
 }

void InfEngineNgraphNet::dfs(std::shared_ptr<ngraph::Node>& node,
                             std::vector<std::shared_ptr<ngraph::Node>>& comp,
                             std::unordered_map<std::string, bool>& used) {
    used[node->get_friendly_name()] = true;
    comp.push_back(node);
    auto inputs = node->get_users();
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        inputs.push_back(node->input_value(i).get_node()->shared_from_this());
    }

    for (auto& to : inputs) {
        if (!used[to->get_friendly_name()]) {
            dfs(to, comp, used);
        }
    }
}

int InfEngineNgraphNet::getNumComponents() {
    if (!components.empty()) {
        return components.size();
    }
    std::unordered_map<std::string, bool> used;
    auto inputs = ngraph_function->get_ordered_ops();
    for (auto& node : inputs) {
        used.emplace(node->get_friendly_name(), false);
    }

    for (auto& node : inputs) {
        if (!used[node->get_friendly_name()]) {
            std::vector<std::shared_ptr<ngraph::Node>> current_comp;
            dfs(node, current_comp, used);
            components.push_back(current_comp);
        }
    }
    return components.size();
}

void InfEngineNgraphNet::createNet(int targetId) {
    if (!hasNetOwner)
    {
        CV_Assert(!unconnectedNodes.empty());
        ngraph::ResultVector outs;
        for (auto& node : unconnectedNodes)
        {
            auto out = std::make_shared<ngraph::op::Result>(node);
            outs.push_back(out);
        }
        CV_Assert_N(!inputs_vec.empty(), !outs.empty());
        ngraph_function = std::make_shared<ngraph::Function>(outs, inputs_vec);

        int num_comp = getNumComponents();
        if (num_comp > 1) {
            for (int i = num_comp - 1; i >= 0; --i) {
                ngraph::ResultVector outputs;
                ngraph::ParameterVector inps;
                for (auto& node : components.back()) {
                    if (node->is_parameter()) {
                        auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(node);
                        inps.push_back(parameter);
                    }
                    else if (node->is_output()) {
                        auto result = std::dynamic_pointer_cast<ngraph::op::Result>(node);
                        outputs.push_back(result);
                    }
                }
                isInit = false;
                CV_Assert_N(!inps.empty(), !outputs.empty());
                ngraph_function = std::make_shared<ngraph::Function>(outputs, inps);
                release();
                components.pop_back();
                init(targetId);
            }
        } else {
            release();
            components.clear();
            init(targetId);
        }
    }
}

void InfEngineNgraphNet::init(int targetId)
{
    if (!hasNetOwner)
    {
        cnn = InferenceEngine::CNNNetwork(InferenceEngine::convertFunctionToICNNNetwork(ngraph_function));
    }

    switch (targetId)
    {
        case DNN_TARGET_CPU:
            device_name = "CPU";
            break;
        case DNN_TARGET_OPENCL:
            device_name = "GPU";
            break;
        case DNN_TARGET_OPENCL_FP16:
            device_name = "GPU";
            // cnn.setPrecision(InferenceEngine::Precision::FP16);
            break;
        case DNN_TARGET_MYRIAD:
            device_name = "MYRIAD";
            // cnn.setPrecision(InferenceEngine::Precision::FP16);
            break;
        case DNN_TARGET_FPGA:
            device_name = "FPGA";
            break;
        default:
            CV_Error(Error::StsNotImplemented, "Unknown target");
    };

    if (!hasNetOwner) {
        for (size_t i = 0; i < ngraph_function->get_output_size(); ++i) {
            auto node = ngraph_function->output(i).get_node();
            for (size_t j = 0; j < node->get_input_size(); ++j) {
                std::string name = node->input_value(j).get_node()->get_friendly_name();
                auto iter = std::find(requestedOutputs.begin(), requestedOutputs.end(), name);
                if (iter != requestedOutputs.end()) {
                    requestedOutputs.erase(iter);
                    cnn.addOutput(name);
                }
            }
        }
    } else {
        for (const auto& name : requestedOutputs)
        {
            cnn.addOutput(name);
        }
    }

    for (const auto& it : cnn.getInputsInfo())
    {
        const std::string& name = it.first;
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        it.second->setPrecision(blobIt->second->getTensorDesc().getPrecision());
    }

    for (const auto& it : cnn.getOutputsInfo())
    {
        const std::string& name = it.first;
        auto blobIt = allBlobs.find(name);
        CV_Assert(blobIt != allBlobs.end());
        it.second->setPrecision(blobIt->second->getTensorDesc().getPrecision());  // Should be always FP32
    }
    // cnn.serialize("/tmp/cnn.xml", "/tmp/cnn.bin");

    initPlugin(cnn);
}

ngraph::ParameterVector InfEngineNgraphNet::setInputs(const std::vector<cv::Mat>& inputs,
                                   const std::vector<std::string>& names) {
    CV_Assert_N(inputs.size() == names.size());
    ngraph::ParameterVector current_inp;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        std::vector<size_t> shape(&inputs[i].size[0], &inputs[i].size[0] + inputs[i].dims);
        auto type = ngraph::element::f32;
        if (inputs[i].type() == CV_16S) {
            type = ngraph::element::f16 ;
        } else if (inputs[i].type() == CV_8U) {
            type = ngraph::element::u8;
        }

        auto inp = std::make_shared<ngraph::op::Parameter>(type, ngraph::Shape(shape));
        inp->set_friendly_name(names[i]);

        auto it = std::find_if(inputs_vec.begin(), inputs_vec.end(),
                                [&inp](const std::shared_ptr<ngraph::op::Parameter>& a) {
                                return a->get_friendly_name() == inp->get_friendly_name();
                  });
        if (it == inputs_vec.end()) {
            inputs_vec.push_back(inp);
            current_inp.push_back(inp);
        } else {
            current_inp.push_back(*it);
        }
    }
    return current_inp;
}

void InfEngineNgraphNet::setUnconnectedNodes(Ptr<InfEngineNgraphNode>& node) {
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
    CV_Assert(!isInitialized());  // some nets
    // net.serialize("/tmp/icnn.xml", "/tmp/icnn.bin");

    try
    {
        AutoLock lock(getInitializationMutex());
        {
            isInit = true;
            // Some of networks can work without a library of extra layers.
#ifndef _WIN32
            // Limit the number of CPU threads.
            if (device_name == "CPU")
                getCore().SetConfig({{
                    InferenceEngine::PluginConfigParams::KEY_CPU_THREADS_NUM, format("%d", getNumThreads()),
                }}, "CPU");
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

InferenceEngine::Blob::Ptr copyBlob(const InferenceEngine::Blob::Ptr& blob)
{
    InferenceEngine::Blob::Ptr copy;
    auto description = blob->getTensorDesc();
    InferenceEngine::Precision precision = description.getPrecision();
    if (precision == InferenceEngine::Precision::FP32)
    {
        copy = InferenceEngine::make_shared_blob<float>(description);
    }
    else if (precision == InferenceEngine::Precision::U8)
    {
        copy = InferenceEngine::make_shared_blob<uint8_t>(description);
    }
    else
        CV_Error(Error::StsNotImplemented, "Unsupported blob precision");
    copy->allocate();
    return copy;
}

InferenceEngine::DataPtr ngraphDataNode(const Ptr<BackendWrapper>& ptr)
{
    CV_Assert(!ptr.empty());
    Ptr<NgraphBackendWrapper> p = ptr.dynamicCast<NgraphBackendWrapper>();
    CV_Assert(!p.empty());
    return p->dataPtr;
}


void forwardNgraph(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                      Ptr<BackendNode>& node, bool isAsync)
{
#ifdef HAVE_INF_ENGINE
    CV_Assert(!node.empty());
    Ptr<InfEngineNgraphNode> ieNode = node.dynamicCast<InfEngineNgraphNode>();
    CV_Assert(!ieNode.empty());
    ieNode->net->forward(outBlobsWrappers, isAsync);
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

void InfEngineNgraphNet::NgraphReqWrapper::makePromises(const std::vector<Ptr<BackendWrapper> >& outsWrappers)
{
    auto outs = ngraphWrappers(outsWrappers);
    outProms.clear();
    outProms.resize(outs.size());
    outsNames.resize(outs.size());
    for (int i = 0; i < outs.size(); ++i)
    {
        outs[i]->futureMat = outProms[i].getArrayResult();
        outsNames[i] = outs[i]->dataPtr->getName();
    }
}

Mat ngraphBlobToMat(const InferenceEngine::Blob::Ptr& blob)
{
    std::vector<size_t> dims = blob->getTensorDesc().getDims();
    std::vector<int> size(dims.begin(), dims.end());
    auto precision = blob->getTensorDesc().getPrecision();

    int type = -1;
    switch (precision)
    {
        case InferenceEngine::Precision::FP32: type = CV_32F; break;
        case InferenceEngine::Precision::U8: type = CV_8U; break;
        default:
            CV_Error(Error::StsNotImplemented, "Unsupported blob precision");
    }
    return Mat(size, type, (void*)blob->buffer());
}

void InfEngineNgraphNet::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync)
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
            inpBlobs[name] = isAsync ? copyBlob(blobIt->second) : blobIt->second;
        }
        for (const auto& it : cnn.getOutputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            CV_Assert(blobIt != allBlobs.end());
            outBlobs[name] = isAsync ? copyBlob(blobIt->second) : blobIt->second;
        }
        reqWrapper->req.SetInput(inpBlobs);
        reqWrapper->req.SetOutput(outBlobs);

        InferenceEngine::IInferRequest::Ptr infRequestPtr = reqWrapper->req;
        infRequestPtr->SetUserData(reqWrapper.get(), 0);

        infRequestPtr->SetCompletionCallback(
            [](InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode status)
            {
                NgraphReqWrapper* wrapper;
                request->GetUserData((void**)&wrapper, 0);
                CV_Assert(wrapper && "Internal error");

                size_t processedOutputs = 0;
                try
                {
                    for (; processedOutputs < wrapper->outProms.size(); ++processedOutputs)
                    {
                        const std::string& name = wrapper->outsNames[processedOutputs];
                        Mat m = ngraphBlobToMat(wrapper->req.GetBlob(name));

                        try
                        {
                            CV_Assert(status == InferenceEngine::StatusCode::OK);
                            wrapper->outProms[processedOutputs].setValue(m.clone());
                        }
                        catch (...)
                        {
                            try {
                                wrapper->outProms[processedOutputs].setException(std::current_exception());
                            } catch(...) {
                                CV_LOG_ERROR(NULL, "DNN: Exception occured during async inference exception propagation");
                            }
                        }
                    }
                }
                catch (...)
                {
                    std::exception_ptr e = std::current_exception();
                    for (; processedOutputs < wrapper->outProms.size(); ++processedOutputs)
                    {
                        try {
                            wrapper->outProms[processedOutputs].setException(e);
                        } catch(...) {
                            CV_LOG_ERROR(NULL, "DNN: Exception occured during async inference exception propagation");
                        }
                    }
                }
                wrapper->isReady = true;
            }
        );
    }

    if (isAsync)
    {
        // Copy actual data to infer request's input blobs.
        for (const auto& it : cnn.getInputsInfo())
        {
            const std::string& name = it.first;
            auto blobIt = allBlobs.find(name);
            Mat srcMat = ngraphBlobToMat(blobIt->second);
            Mat dstMat = ngraphBlobToMat(reqWrapper->req.GetBlob(name));
            srcMat.copyTo(dstMat);
        }

        // Set promises to output blobs wrappers.
        reqWrapper->makePromises(outBlobsWrappers);

        reqWrapper->isReady = false;
        reqWrapper->req.StartAsync();
    }
    else
    {
        reqWrapper->req.Infer();
    }
}

#endif

}}
