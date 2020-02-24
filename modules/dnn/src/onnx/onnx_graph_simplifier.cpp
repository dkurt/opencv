// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#include "../graph_simplifier.hpp"
#include "onnx_graph_simplifier.hpp"

#include <queue>

namespace cv { namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

// This wrapper can behave differently for fake input nodes and real graph nodes.
class ONNXNodeWrapper : public ImportNodeWrapper
{
public:
    ONNXNodeWrapper(opencv_onnx::NodeProto* _node = 0) : node(_node) {}

    virtual int getNumInputs() const CV_OVERRIDE
    {
        return node ? node->input_size() : 0;
    }

    virtual std::string getInputName(int idx) const CV_OVERRIDE
    {
        CV_Assert_N(node, idx < node->input_size());
        return node->input(idx);
    }

    virtual std::string getType() const CV_OVERRIDE
    {
        return node ? node->op_type() : "";
    }

    virtual void setType(const std::string& type) CV_OVERRIDE
    {
        CV_Assert(node);
        node->set_op_type(type);
    }

    virtual void setInputNames(const std::vector<std::string>& inputs) CV_OVERRIDE
    {
        CV_Assert(node);
        node->clear_input();
        for (int i = 0; i < inputs.size(); ++i)
            node->add_input(inputs[i]);
    }

    opencv_onnx::NodeProto* node;
};

// ONNX graph's inputs are separate from nodes so we index them before the rest of nodes.
class ONNXGraphWrapper : public ImportGraphWrapper
{
public:
    ONNXGraphWrapper(opencv_onnx::GraphProto& _net) : net(_net)
    {
        numInputs = net.input_size();
    }

    virtual Ptr<ImportNodeWrapper> getNode(int idx) const CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = 0;
        if (idx >= numInputs)
            node = net.mutable_node(idx - numInputs);
        return makePtr<ONNXNodeWrapper>(node);
    }

    virtual int getNumNodes() const CV_OVERRIDE
    {
        return numInputs + net.node_size();
    }

    virtual int getNumOutputs(int nodeId) const CV_OVERRIDE
    {
        if (nodeId < numInputs)
            return 1;
        else
            return net.node(nodeId - numInputs).output_size();
    }

    virtual std::string getOutputName(int nodeId, int outId) const CV_OVERRIDE
    {
        CV_Assert(outId < getNumOutputs(nodeId));
        if (nodeId < numInputs)
            return net.input(nodeId).name();
        else
            return net.node(nodeId - numInputs).output(outId);
    }

    virtual void removeNode(int idx) CV_OVERRIDE
    {
        CV_Assert(idx >= numInputs);
        net.mutable_node()->DeleteSubrange(idx - numInputs, 1);
    }

private:
    int numInputs;
    opencv_onnx::GraphProto& net;
};

class SoftMaxSubgraph : public Subgraph
{
public:
    SoftMaxSubgraph() : axis(1)
    {
        int input = addNodeToMatch("");
        int inpExp = addNodeToMatch("Exp", input);
        int sum = addNodeToMatch("ReduceSum", inpExp);
        addNodeToMatch("Div", inpExp, sum);
        setFusedNode("Softmax", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (Subgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            Ptr<ImportNodeWrapper> sum = net->getNode(matchedNodesIds[1]);
            opencv_onnx::NodeProto* node = sum.dynamicCast<ONNXNodeWrapper>()->node;

            for (int i = 0; i < node->attribute_size(); i++)
            {
                opencv_onnx::AttributeProto attr = node->attribute(i);
                if (attr.name() != "axes")
                    continue;
                if (attr.ints_size() != 1)
                    CV_Error(Error::StsNotImplemented, format("Unexpected number of axes: %d", attr.ints_size()));
                axis = attr.ints(0);
                return true;
            }
            CV_Error(Error::StsNotImplemented, "Missed axes attribute");
        }
        return false;
    }

    virtual void finalize(const Ptr<ImportGraphWrapper>&,
                          const Ptr<ImportNodeWrapper>& fusedNode,
                          std::vector<Ptr<ImportNodeWrapper> >&) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attr = node->add_attribute();
        attr->set_name("axis");
        attr->set_i(axis);
    }

private:
    int axis;
};

class FuseUpsampleSubgraph : public Subgraph
{
public:
    FuseUpsampleSubgraph() : scaleH(1), scaleW(1), emptyNodeId(-1) {}

    void finalize(const Ptr<ImportGraphWrapper>& net,
                  const Ptr<ImportNodeWrapper>& fusedNode,
                  std::vector<Ptr<ImportNodeWrapper> >& inputs) CV_OVERRIDE
    {
        opencv_onnx::NodeProto* node = fusedNode.dynamicCast<ONNXNodeWrapper>()->node;
        opencv_onnx::AttributeProto* attrH = node->add_attribute();
        attrH->set_name("height_scale");
        attrH->set_i(scaleH);
        opencv_onnx::AttributeProto* attrW = node->add_attribute();
        attrW->set_name("width_scale");
        attrW->set_i(scaleW);
        //remove empty constant node
        if (emptyNodeId != -1)
            net->removeNode(emptyNodeId);
    }

protected:
    int scaleH, scaleW, emptyNodeId;
};

class ExtractScalesSubgraph : public FuseUpsampleSubgraph
{
public:
    ExtractScalesSubgraph()
    {
        int input = addNodeToMatch("");

        int indexH = addNodeToMatch("Constant");
        int shape1 = addNodeToMatch("Shape", input);
        int gather1 = addNodeToMatch("Gather", shape1, indexH);
        int castG1 = addNodeToMatch("Cast", gather1);
        int scaleH = addNodeToMatch("Constant");
        int mul1 = addNodeToMatch("Mul", castG1, scaleH);
        int castM1 = addNodeToMatch("Cast", mul1);
        int floor1 = addNodeToMatch("Floor", castM1);

        int indexW = addNodeToMatch("Constant");
        int shape2 = addNodeToMatch("Shape", input);
        int gather2 = addNodeToMatch("Gather", shape2, indexW);
        int castG2 = addNodeToMatch("Cast", gather2);
        int scaleW = addNodeToMatch("Constant");
        int mul2 = addNodeToMatch("Mul", castG2, scaleW);
        int castM2 = addNodeToMatch("Cast", mul2);
        int floor2 = addNodeToMatch("Floor", castM2);

        int unsqueeze1 = addNodeToMatch("Unsqueeze", floor1);
        int unsqueeze2 = addNodeToMatch("Unsqueeze", floor2);
        addNodeToMatch("Concat", unsqueeze1, unsqueeze2);

        setFusedNode("Scales", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (FuseUpsampleSubgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            std::vector<cv::Mat> nodes_attributes;
            for (int i = matchedNodesIds.front(); i < matchedNodesIds.back(); i++)
            {
                if (std::find(matchedNodesIds.begin(), matchedNodesIds.end(), i) == matchedNodesIds.end())
                {
                    opencv_onnx::NodeProto* constant_node = net->getNode(i).dynamicCast<ONNXNodeWrapper>()->node;
                    opencv_onnx::TensorProto tensor_proto = constant_node->attribute(0).t();
                    nodes_attributes.push_back(getMatFromTensor(tensor_proto));
                }
            }
            CV_Assert(nodes_attributes[0].total() == 1/*indexH*/ && nodes_attributes[1].total() == 1/*scaleH*/ &&
                nodes_attributes[2].total() == 1/*indexW*/ && nodes_attributes[3].total() == 1/*scaleW*/);

            if (nodes_attributes[0].at<int>(0) == 2 && nodes_attributes[2].at<int>(0) == 3)
            {
                scaleH = (int)nodes_attributes[1].at<float>(0);
                scaleW = (int)nodes_attributes[3].at<float>(0);
            }
            else
            {
                scaleH = (int)nodes_attributes[3].at<float>(0);
                scaleW = (int)nodes_attributes[1].at<float>(0);
            }
            return true;
        }
        return false;
    }
};

class UpsampleSubgraph : public FuseUpsampleSubgraph
{
public:
    UpsampleSubgraph()
    {
        int input = addNodeToMatch("");

        int scaleHW = addNodeToMatch("Scales", input);
        int scaleNC = addNodeToMatch("Constant");
        int cast1 = addNodeToMatch("Cast", scaleHW);

        int shape = addNodeToMatch("Shape", input);
        int slice = addNodeToMatch("Slice", shape);
        int cast2 = addNodeToMatch("Cast", slice);

        int inpDiv = addNodeToMatch("Div", cast1, cast2);
        int inpConcat2 = addNodeToMatch("Concat", scaleNC, inpDiv);

        addNodeToMatch("Upsample", input, inpConcat2);
        setFusedNode("Upsample", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (FuseUpsampleSubgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            Ptr<ImportNodeWrapper> scales = net->getNode(matchedNodesIds[0]);
            opencv_onnx::NodeProto* node = scales.dynamicCast<ONNXNodeWrapper>()->node;
            opencv_onnx::AttributeProto attrH = node->attribute(1);
            scaleH = attrH.i();
            opencv_onnx::AttributeProto attrW = node->attribute(2);
            scaleW = attrW.i();
            return true;
        }
        return false;
    }
};

class ResizeSubgraph : public FuseUpsampleSubgraph
{
public:
    ResizeSubgraph()
    {
        int input = addNodeToMatch("");

        int scaleHW = addNodeToMatch("Scales", input);
        int constantROI = addNodeToMatch("Constant");

        int shape = addNodeToMatch("Shape", input);
        int axes_slice = addNodeToMatch("Constant");
        int start_slice = addNodeToMatch("Constant");
        int end_slice = addNodeToMatch("Constant");
        int slice = addNodeToMatch("Slice", shape, start_slice, end_slice, axes_slice);

        int cast = addNodeToMatch("Cast", scaleHW);

        int concat = addNodeToMatch("Concat", slice, cast);
        addNodeToMatch("Resize", input, constantROI, constantROI, concat);
        setFusedNode("Upsample", input);
    }

    virtual bool match(const Ptr<ImportGraphWrapper>& net, int nodeId,
                       std::vector<int>& matchedNodesIds,
                       std::vector<int>& targetNodesIds) CV_OVERRIDE
    {
        if (FuseUpsampleSubgraph::match(net, nodeId, matchedNodesIds, targetNodesIds))
        {
            Ptr<ImportNodeWrapper> scales = net->getNode(matchedNodesIds[0]);
            opencv_onnx::NodeProto* node = scales.dynamicCast<ONNXNodeWrapper>()->node;
            opencv_onnx::AttributeProto attr = node->attribute(1);
            opencv_onnx::AttributeProto attrH = node->attribute(1);
            scaleH = attrH.i();
            opencv_onnx::AttributeProto attrW = node->attribute(2);
            scaleW = attrW.i();
            //constant node id after matchedNodesIds removal.
            emptyNodeId = matchedNodesIds.front();
            return true;
        }
        return false;
    }
};

void simplifySubgraphs(opencv_onnx::GraphProto& net)
{
    std::vector<Ptr<Subgraph> > subgraphs;
    subgraphs.push_back(makePtr<ExtractScalesSubgraph>());
    subgraphs.push_back(makePtr<UpsampleSubgraph>());
    subgraphs.push_back(makePtr<ResizeSubgraph>());
    subgraphs.push_back(makePtr<SoftMaxSubgraph>());

    simplifySubgraphs(Ptr<ImportGraphWrapper>(new ONNXGraphWrapper(net)), subgraphs);
}

Mat getMatFromTensor(opencv_onnx::TensorProto& tensor_proto)
{
    CV_Assert(!tensor_proto.raw_data().empty() || !tensor_proto.float_data().empty()
                    || !tensor_proto.double_data().empty() || !tensor_proto.int64_data().empty());

    opencv_onnx::TensorProto_DataType datatype = tensor_proto.data_type();
    Mat blob;
    std::vector<int> sizes;
    for (int i = 0; i < tensor_proto.dims_size(); i++) {
            sizes.push_back(tensor_proto.dims(i));
    }
    if (sizes.empty())
        sizes.assign(1, 1);
    if (datatype == opencv_onnx::TensorProto_DataType_FLOAT) {

        if (!tensor_proto.float_data().empty()) {
            const ::google::protobuf::RepeatedField<float> field = tensor_proto.float_data();
            Mat(sizes, CV_32FC1, (void*)field.data()).copyTo(blob);
        }
        else {
            char* val = const_cast<char*>(tensor_proto.raw_data().c_str());
            Mat(sizes, CV_32FC1, val).copyTo(blob);
        }
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_DOUBLE)
    {
        const ::google::protobuf::RepeatedField<double> field = tensor_proto.double_data();
        CV_Assert(!field.empty());
        Mat(sizes, CV_64FC1, (void*)field.data()).convertTo(blob, CV_32FC1);
    }
    else if (datatype == opencv_onnx::TensorProto_DataType_INT64)
    {
        blob.create(sizes, CV_32SC1);
        int32_t* dst = reinterpret_cast<int32_t*>(blob.data);

        if (!tensor_proto.int64_data().empty()) {
            ::google::protobuf::RepeatedField< ::google::protobuf::int64> src = tensor_proto.int64_data();
            convertInt64ToInt32(src, dst, blob.total());
        }
        else
        {
            const char* val = tensor_proto.raw_data().c_str();
#if CV_STRONG_ALIGNMENT
            // Aligned pointer is required: https://github.com/opencv/opencv/issues/16373
            // this doesn't work: typedef int64_t CV_DECL_ALIGNED(1) unaligned_int64_t;
            AutoBuffer<int64_t, 16> aligned_val;
            if (!isAligned<sizeof(int64_t)>(val))
            {
                size_t sz = tensor_proto.raw_data().size();
                aligned_val.allocate(divUp(sz, sizeof(int64_t)));
                memcpy(aligned_val.data(), val, sz);
                val = (const char*)aligned_val.data();
            }
#endif
            const int64_t* src = reinterpret_cast<const int64_t*>(val);
            convertInt64ToInt32(src, dst, blob.total());
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "Unsupported data type: " +
                        opencv_onnx::TensorProto_DataType_Name(datatype));
    if (tensor_proto.dims_size() == 0)
        blob.dims = 1;  // To force 1-dimensional cv::Mat for scalars.
    return blob;
}

CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace cv::dnn
