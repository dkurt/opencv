// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF
#include "tf_graph_simplifier.hpp"
#include <fstream>

namespace cv { namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

using google::protobuf::RepeatedPtrField;
using google::protobuf::MapPair;
using google::protobuf::TextFormat;

// Split source text by tokens.
// Delimeters are specific for protobuf in text format.
static void tokenize(const std::string& str, std::vector<std::string>& tokens)
{
    tokens.clear();
    tokens.reserve(max(1, (int)str.size() / 7));

    std::string token = "";
    bool isString = false;  // Flag to manage empty strings.
    bool isComment = false;
    for (size_t i = 0, n = str.size(); i < n; ++i)
    {
        char symbol = str[i];
        isComment = (isComment && symbol != '\n') || (!isString && symbol == '#');
        if (isComment)
            continue;

        if (symbol == ' ' || symbol == '\t' || symbol == '\r' || symbol == '\'' ||
            symbol == '\n' || symbol == ':' || symbol == '\"' || symbol == ';' ||
            symbol == ',')
        {
            if (!token.empty() || ((symbol == '\"' || symbol == '\'') && isString))
            {
                tokens.push_back(token);
                token = "";
            }
            isString = (symbol == '\"' || symbol == '\'') ^ isString;
        }
        else if (symbol == '{' || symbol == '}' || symbol == '[' || symbol == ']')
        {
            if (!token.empty())
            {
                tokens.push_back(token);
                token = "";
            }
            tokens.push_back(std::string(1, symbol));
        }
        else
        {
            token += symbol;
        }
    }
    if (!token.empty())
    {
        tokens.push_back(token);
    }
}

// A minimal structure for parsing text protobuf messages.
struct TextMessage
{
    std::map<std::string, std::vector<std::string> > fields;
    std::map<std::string, Ptr<TextMessage> > messages;
};

struct TextMessageNode
{
public:
      TextMessageNode(Ptr<TextMessage> msg_) : msg(msg_) {}

      TextMessageNode(const std::vector<std::string>& values_) : values(values_) {}

      TextMessageNode(const std::string& value) : values(1, value) {}

      TextMessageNode operator[](const std::string& name) const
      {
          std::map<std::string, Ptr<TextMessage> >::iterator it = msg->messages.find(name);
          if (it != msg->messages.end())
              return TextMessageNode(it->second);

          std::map<std::string, std::vector<std::string> >::iterator fieldIt = msg->fields.find(name);
          if (fieldIt != msg->fields.end())
          {
              return TextMessageNode(fieldIt->second);
          }
          CV_Error(Error::StsObjectNotFound, "An entry with name " + name + " not found");
      }

      TextMessageNode operator[](const char* name) const
      {
          return operator[](std::string(name));
      }

      TextMessageNode operator[](int idx) const
      {
          CV_Assert(msg.empty());
          CV_Assert(idx < values.size());
          return TextMessageNode(values[idx]);
      }

      size_t size() const
      {
          CV_Assert(msg.empty());
          return values.size();
      }

      operator std::string() const
      {
          CV_Assert(msg.empty());
          CV_Assert(values.size() == 1);
          return values[0];
      }

      operator int32_t() const { return getValue<int32_t>(); }
      operator float()   const { return getValue<float>(); }
      operator bool()    const { return getValue<bool>(); }

      bool has(const std::string& entry) const
      {
          return values.empty() &&
                 (msg->messages.find(entry) != msg->messages.end() ||
                  msg->fields.find(entry) != msg->fields.end());
      }

private:
      template <typename T>
      T getValue() const
      {
          CV_Assert(msg.empty());
          CV_Assert(values.size() == 1);

          T value = 0;
          std::string str = values[0];
          if (!str.empty())
          {
              if (typeid(T) != typeid(bool))
              {
                  std::stringstream ss(str);
                  ss >> value;
              }
              else if (str == "true")
              {
                  memset(&value, true, 1);
              }
              else if (str == "false")
              {
                  memset(&value, false, 1);
              }
              else
              {
                  CV_Error(Error::StsParseError,
                           "Cannot interpret boolean value: " + str);
              }
          }
          return value;
      }

      Ptr<TextMessage> msg;
      std::vector<std::string> values;
};

static Ptr<TextMessage> parseTextMessage(std::vector<std::string>::iterator& tokenIt)
{
    Ptr<TextMessage> msg(new TextMessage());
    CV_Assert(*tokenIt == "{");

    std::string fieldName, fieldValue;
    bool isArray = false;
    for (;;)
    {
        if (!isArray)
        {
            ++tokenIt;
            fieldName = *tokenIt;

            if (fieldName == "}")
                break;
        }

        ++tokenIt;
        fieldValue = *tokenIt;

        if (fieldValue == "{")
            msg->messages.insert(std::make_pair(fieldName, parseTextMessage(tokenIt)));
        else if (fieldValue == "[")
            isArray = true;
        else if (fieldValue == "]")
            isArray = false;
        else
        {
            std::map<std::string, std::vector<std::string> >::iterator it = msg->fields.find(fieldName);
            if (it == msg->fields.end())
                msg->fields.insert(std::make_pair(fieldName, std::vector<std::string>(1, fieldValue)));
            else
                it->second.push_back(fieldValue);
        }
    }
    return msg;
}

template <typename T>
static std::string tensorMsg(const std::vector<T>& values)
{
    std::string dtype, field;
    if (typeid(T) == typeid(float))
    {
        dtype = "DT_FLOAT";
        field = "float_val";
    }
    else if (typeid(T) == typeid(int32_t))
    {
        dtype = "DT_INT32";
        field = "int_val";
    }
    else
        CV_Error(Error::StsNotImplemented, "Unsupported tensor type");

    std::string msg = "tensor { dtype: " + dtype + cv::format(" tensor_shape { dim { size: %d } }", values.size());
    for (int i = 0; i < values.size(); ++i)
    {
        std::stringstream ss;
        ss << field << ": " << values[i] << " ";
        msg += ss.str();
    }
    return msg + "}";
}

template <typename T>
static void addConstNode(const std::string& name, const std::vector<T>& values, tensorflow::GraphDef& net)
{
    tensorflow::NodeDef* node = net.add_node();
    node->set_name(name);
    node->set_op("Const");

    tensorflow::AttrValue value;
    TextFormat::MergeFromString(tensorMsg(values), &value);
    node->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>("value", value));
}

template <typename T>
static void addConstNode(const std::string& name, T value, tensorflow::GraphDef& net)
{
    std::vector<T> values(1, value);
    addConstNode(name, values, net);
}

static void addConcatNode(const std::string& name,
                          const std::vector<std::string>& inputs,
                          const std::string& axisNodeName,
                          tensorflow::GraphDef& net)
{
    tensorflow::NodeDef* node = net.add_node();
    node->set_name(name);
    node->set_op("ConcatV2");
    for (int i = 0; i < inputs.size(); ++i)
        node->add_input(inputs[i]);
    node->add_input(axisNodeName);
}

static void addReshapeNode(const std::string& name, const std::string& input,
                           const std::vector<int>& newShape,
                           tensorflow::GraphDef& net)
{
    addConstNode(name + "/shape", newShape, net);

    tensorflow::NodeDef* reshape = net.add_node();
    reshape->set_name(name);
    reshape->set_op("Reshape");
    reshape->add_input(input);
    reshape->add_input(name + "/shape");
}

template <typename T>
static void addAttr(const std::string& name, T value, tensorflow::NodeDef* node)
{
    tensorflow::AttrValue attr;
    if (typeid(T) == typeid(float))
    {
        attr.set_f(value);
    }
    else if (typeid(T) == typeid(bool))
    {
        attr.set_b(value);
    }
    else if (typeid(T) == typeid(int32_t))
    {
        attr.set_i(value);
    }
    else
        CV_Error(Error::StsNotImplemented, "Unsupported type of value " + name);
    node->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>(name, attr));
}

template <typename >
static void addAttr(const std::string& name, const std::string& value, tensorflow::NodeDef* node)
{
    tensorflow::AttrValue attr;
    attr.set_s(value);
    node->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>(name, attr));
}

template <typename T>
static void addAttr(const std::string& name, const std::vector<T>& values,
                    tensorflow::NodeDef* node)
{
    tensorflow::AttrValue attr;
    TextFormat::MergeFromString(tensorMsg(values), &attr);
    node->mutable_attr()->insert(MapPair<std::string, tensorflow::AttrValue>(name, attr));
}

static void addSoftmaxNode(const std::string& name, const std::string& input,
                           tensorflow::GraphDef& net)
{
    tensorflow::NodeDef* node = net.add_node();
    node->set_name(name);
    node->set_op("Softmax");
    addAttr("axis", -1, node);
    node->add_input(input);
}

def addSlice(inp, out, begins, sizes):
    beginsNode = NodeDef()
    beginsNode.name = out + '/begins'
    beginsNode.op = 'Const'
    text_format.Merge(tensorMsg(begins), beginsNode.attr["value"])
    graph_def.node.extend([beginsNode])

    sizesNode = NodeDef()
    sizesNode.name = out + '/sizes'
    sizesNode.op = 'Const'
    text_format.Merge(tensorMsg(sizes), sizesNode.attr["value"])
    graph_def.node.extend([sizesNode])

    sliced = NodeDef()
    sliced.name = out
    sliced.op = 'Slice'
    sliced.input.append(inp)
    sliced.input.append(beginsNode.name)
    sliced.input.append(sizesNode.name)
    graph_def.node.extend([sliced])


static void addDetectionOutputNode(const std::string& name, const std::string& inpBoxes,
                                   const std::string& inpScores, const std::string& inpPriors,
                                   int numClasses, float nmsThresh, float scoreThresh,
                                   int top_k, tensorflow::GraphDef& net)
{
    tensorflow::NodeDef* node = net.add_node();
    node->set_name("detection_out");
    node->set_op("DetectionOutput");

    node->add_input(inpBoxes);
    node->add_input(inpScores);
    node->add_input(inpPriors);

    addAttr<int>("num_classes", numClasses, node);
    addAttr<bool>("share_location", true, node);
    addAttr<int>("background_label_id", 0, node);
    addAttr<float>("nms_threshold", nmsThresh, node);
    addAttr<int>("top_k", top_k, node);
    addAttr<std::string>("code_type", "CENTER_SIZE", node);
    addAttr<int>("keep_top_k", 100, node);
    addAttr<float>("confidence_threshold", scoreThresh, node);
}

static tensorflow::NodeDef* addNode(const std::string& type, const std::string& name,
                                    const std::string& input, tensorflow::GraphDef& net)
{
    tensorflow::NodeDef* node = net.add_node();
    node->set_name(name);
    node->set_op(type);
    node->add_input(input);
    return node;
}

class ObjectDetectionSimplifier
{
protected:
    virtual bool toRemove(const tensorflow::NodeDef& node) = 0;

    virtual void process(tensorflow::GraphDef& net, const TextMessageNode& config) = 0;

    // Remove unused nodes.
    void removeUnusedNodes(tensorflow::GraphDef& net)
    {
        std::vector<std::string> removedNodes;
        RepeatedPtrField<tensorflow::NodeDef>::iterator it;
        google::protobuf::Map<std::string, tensorflow::AttrValue>::iterator attrIt;
        for (it = net.mutable_node()->begin(); it != net.mutable_node()->end();)
        {
            tensorflow::NodeDef& layer = *it;
            std::string op = layer.op();
            std::string name = layer.name();

            if (toRemove(layer))
            {
                removedNodes.push_back(name);
                it = net.mutable_node()->erase(it);
            }
            else
                ++it;
        }

        // Remove references to removed nodes.
        RepeatedPtrField<std::string>::iterator inputIt;
        for (it = net.mutable_node()->begin(); it != net.mutable_node()->end(); ++it)
        {
            tensorflow::NodeDef& layer = *it;
            for (inputIt = layer.mutable_input()->begin(); inputIt != layer.mutable_input()->end();)
            {
                if (std::find(removedNodes.begin(), removedNodes.end(), *inputIt) != removedNodes.end())
                    inputIt = layer.mutable_input()->erase(inputIt);
                else
                    ++inputIt;
            }
        }
    }

    void removeUnconnectedNodes(tensorflow::GraphDef& net, const std::string& excludeName)
    {
        std::vector<std::string> unconnected;
        RepeatedPtrField<tensorflow::NodeDef>::iterator it;
        for (;;)
        {
            getUnconnectedNodes(net, unconnected);
            unconnected.erase(std::remove(unconnected.begin(), unconnected.end(), excludeName),
                              unconnected.end());

            if (unconnected.empty())
                break;

            for (int i = 0; i < unconnected.size(); ++i)
            {
                for (it = net.mutable_node()->begin(); it != net.mutable_node()->end(); ++it)
                {
                    if (it->name() == unconnected[i])
                    {
                        net.mutable_node()->erase(it);
                        break;
                    }
                }
            }
        }
    }

private:
    static void getUnconnectedNodes(const tensorflow::GraphDef& net,
                                    std::vector<std::string>& names)
    {
        names.clear();
        for (int i = 0; i < net.node_size(); ++i)
        {
            const tensorflow::NodeDef& node = net.node(i);
            names.push_back(node.name());
            for (int j = 0; j < node.input_size(); ++j)
            {
                names.erase(std::remove(names.begin(), names.end(), node.input(j)), names.end());
            }
        }
    }
};

class FasterRCNNSimplifier CV_FINAL : public ObjectDetectionSimplifier
{
public:
    FasterRCNNSimplifier()
    {
        scopesToKeep.push_back("FirstStageFeatureExtractor");
        scopesToKeep.push_back("Conv");
        scopesToKeep.push_back("FirstStageBoxPredictor/BoxEncodingPredictor");
        scopesToKeep.push_back("FirstStageBoxPredictor/ClassPredictor");
        scopesToKeep.push_back("CropAndResize");
        scopesToKeep.push_back("MaxPool2D");
        scopesToKeep.push_back("SecondStageFeatureExtractor");
        scopesToKeep.push_back("SecondStageBoxPredictor");
        scopesToKeep.push_back("image_tensor");

        scopesToIgnore.push_back("FirstStageFeatureExtractor/Assert");
        scopesToIgnore.push_back("FirstStageFeatureExtractor/Shape");
        scopesToIgnore.push_back("FirstStageFeatureExtractor/strided_slice");
        scopesToIgnore.push_back("FirstStageFeatureExtractor/GreaterEqual");
        scopesToIgnore.push_back("FirstStageFeatureExtractor/LogicalAnd");
    }

    virtual void process(tensorflow::GraphDef& net, const TextMessageNode& config) CV_OVERRIDE
    {
        // TextMessageNode ssdAnchorGenerator = ;
        TextMessageNode aspectRatios = config["first_stage_anchor_generator"]["grid_anchor_generator"]["aspect_ratios"];
        TextMessageNode scales = config["first_stage_anchor_generator"]["grid_anchor_generator"]["scales"];

        sortByExecutionOrder(net);
        simplifySubgraphs(net);
        RemoveIdentityOps(net);

        removeUnusedNodes(net);

        std::string netInputName;
        for (int i = 0; i < net.node_size(); ++i)
        {
            const tensorflow::NodeDef& node = net.node(i);
            if (node.op() == "Placeholder")
            {
                CV_Assert(i < net.node_size() - 1);
                netInputName = node.name();

                tensorflow::NodeDef* consumer = 0;
                for (int j = i + 1; j < net.node_size(); ++j)
                {
                    consumer = net.mutable_node(j);
                    if (consumer->op() != "Const")
                        break;
                }
                CV_Assert(consumer);

                std::string weights = consumer->input(0);  // Convolution weights
                consumer->clear_input();
                consumer->add_input(netInputName);
                consumer->add_input(weights);
                break;
            }
        }
        CV_Assert(!netInputName.empty());

        // Temporarily remove top nodes.
        std::vector<tensorflow::NodeDef*> topNodes;
        while (net.node_size())
        {
            tensorflow::NodeDef* topNode = net.mutable_node()->ReleaseLast();
            topNodes.push_back(topNode);
            std::cout << topNode->name() << '\n';
            if (topNode->op() == "CropAndResize")
                break;
        }

        std::vector<int> shape(3);
        shape[0] = 0; shape[1] = -1; shape[2] = 2;
        addReshapeNode("FirstStageBoxPredictor/ClassPredictor/reshape_1",
                       "FirstStageBoxPredictor/ClassPredictor/BiasAdd", shape, net);
        addSoftmaxNode("FirstStageBoxPredictor/ClassPredictor/softmax",
                       "FirstStageBoxPredictor/ClassPredictor/reshape_1", net);
        addNode("Flatten", "FirstStageBoxPredictor/ClassPredictor/softmax/flatten",
                "FirstStageBoxPredictor/ClassPredictor/softmax", net);
        addNode("Flatten", "FirstStageBoxPredictor/BoxEncodingPredictor/flatten",
                "FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd", net);

        // Proposals generator.
        tensorflow::NodeDef* priorBox = net.add_node();
        priorBox->set_name("proposals");  // Compare with ClipToWindow/Gather/Gather (NOTE: normalized)
        priorBox->set_op("PriorBox");
        priorBox->add_input("FirstStageBoxPredictor/BoxEncodingPredictor/BiasAdd");
        priorBox->add_input(netInputName);

        addAttr<bool>("flip", false, priorBox);
        addAttr<bool>("clip", true, priorBox);
        addAttr<float>("step", 16.0f, priorBox);
        addAttr<float>("offset", 0.0f, priorBox);

        std::vector<float> widths, heights;
        for (int i = 0; i < aspectRatios.size(); ++i)
        {
            float ar = std::sqrt((float)aspectRatios[i]);
            float features_stride = 16;
            float features_stride_sq = 16*16;
            for (int j = 0; j < scales.size(); ++j)
            {
                float s = scales[j];
                heights.push_back(features_stride_sq * s / ar);
                widths.push_back(features_stride_sq * s * ar);
            }
        }
        addAttr("width", widths, priorBox);
        addAttr("height", heights, priorBox);
        std::vector<float> variance(4);
        variance[0] = variance[1] = 0.1;
        variance[2] = variance[3] = 0.2;
        addAttr("variance", variance, priorBox);

        addDetectionOutputNode("detection_out",
                               "FirstStageBoxPredictor/BoxEncodingPredictor/flatten",
                               "FirstStageBoxPredictor/ClassPredictor/softmax/flatten",
                               "proposals", 2, 0.7, 0.0, 6000, net);

        addConstNode<int>("clip_by_value/lower", 0.0, net);
        addConstNode<int>("clip_by_value/upper", 1.0, net);

        tensorflow::NodeDef* clipByValueNode = net.add_node();
        clipByValueNode->set_name("detection_out/clip_by_value");
        clipByValueNode->set_op("ClipByValue");
        clipByValueNode->add_input("detection_out");
        clipByValueNode->add_input("clip_by_value/lower");
        clipByValueNode->add_input("clip_by_value/upper");

        for (int i = topNodes.size() - 1; i >= 0; --i)
        {
            net.mutable_node()->AddAllocated(topNodes[i]);
        }

        addSoftmaxNode("SecondStageBoxPredictor/Reshape_1/softmax",
                       "SecondStageBoxPredictor/Reshape_1", net);

        addSlice('SecondStageBoxPredictor/Reshape_1/softmax',
                 'SecondStageBoxPredictor/Reshape_1/slice',
                 [0, 0, 1], [-1, -1, -1])

        addReshape('SecondStageBoxPredictor/Reshape_1/slice',
                  'SecondStageBoxPredictor/Reshape_1/Reshape', [1, -1])
    }

private:
    virtual bool toRemove(const tensorflow::NodeDef& node) CV_OVERRIDE
    {
        const std::string& name = node.name();
        for (int i = 0; i < scopesToIgnore.size(); ++i)
        {
            if (name.find(scopesToIgnore[i]) == 0)
                return true;
        }

        bool keep = false;
        for (int i = 0; !keep && i < scopesToKeep.size(); ++i)
        {
            keep = name.find(scopesToKeep[i]) == 0;
        }
        return !keep;
    }

    std::vector<std::string> scopesToKeep, scopesToIgnore;
};

class SSDSimplifier CV_FINAL : public ObjectDetectionSimplifier
{
public:
    SSDSimplifier()
    {
        keepOps.insert("Conv2D");
        keepOps.insert("BiasAdd");
        keepOps.insert("Add");
        keepOps.insert("Relu6");
        keepOps.insert("Placeholder");
        keepOps.insert("FusedBatchNorm");
        keepOps.insert("DepthwiseConv2dNative");
        keepOps.insert("ConcatV2");
        keepOps.insert("Mul");
        keepOps.insert("MaxPool");
        keepOps.insert("AvgPool");
        keepOps.insert("Identity");
        keepOps.insert("Const");
    }

    virtual void process(tensorflow::GraphDef& net, const TextMessageNode& config) CV_OVERRIDE
    {
        const int numClasses = config["num_classes"];
        TextMessageNode ssdAnchorGenerator = config["anchor_generator"]["ssd_anchor_generator"];
        const float minScale = ssdAnchorGenerator["min_scale"];
        const float maxScale = ssdAnchorGenerator["max_scale"];
        const int numLayers = ssdAnchorGenerator["num_layers"];
        const int imgWidth = config["image_resizer"]["fixed_shape_resizer"]["width"];
        const int imgHeight = config["image_resizer"]["fixed_shape_resizer"]["height"];
        TextMessageNode aspectRatios = ssdAnchorGenerator["aspect_ratios"];
        const float scoreThresh = config["post_processing"]["batch_non_max_suppression"]["score_threshold"];
        const float iouThresh = config["post_processing"]["batch_non_max_suppression"]["iou_threshold"];

        bool reduceBoxesInLowestLayer = true;
        if (ssdAnchorGenerator.has("reduce_boxes_in_lowest_layer"))
            reduceBoxesInLowestLayer = (bool)ssdAnchorGenerator["reduce_boxes_in_lowest_layer"];

        sortByExecutionOrder(net);

        std::vector<std::string> boxesPredictionNodes;
        std::vector<std::string> classPredictionNodes;

        std::map<std::string, tensorflow::NodeDef*> nodesMap;
        std::map<std::string, tensorflow::NodeDef*>::iterator nodesMapIt;
        for (int i = 0; i < net.node_size(); ++i)
        {
            tensorflow::NodeDef* node = net.mutable_node(i);
            std::string nodeName = node->name();
            nodesMap[nodeName] = node;

            std::vector<std::string>* dst = 0;
            if (nodeName == "concat")
                dst = &boxesPredictionNodes;
            else if (nodeName == "concat_1")
                dst = &classPredictionNodes;

            if (dst)
            {
                CV_Assert(node->input_size() == numLayers + 1);  // inputs and axis
                for (int j = 0; j < numLayers; ++j)
                {
                    nodesMapIt = nodesMap.find(node->input(j));
                    CV_Assert(nodesMapIt != nodesMap.end());
                    tensorflow::NodeDef* inpNode = nodesMapIt->second;
                    while (inpNode->op() == "Reshape" || inpNode->op() == "Squeeze")
                    {
                        nodesMapIt = nodesMap.find(inpNode->input(0));
                        CV_Assert(nodesMapIt != nodesMap.end());
                        inpNode = nodesMapIt->second;
                    }
                    dst->push_back(inpNode->name().substr(0, inpNode->name().rfind('/')));
                }
            }
        }
        CV_Assert(boxesPredictionNodes.size() == numLayers);
        CV_Assert(classPredictionNodes.size() == numLayers);

        simplifySubgraphs(net);
        RemoveIdentityOps(net);

        removeUnusedNodes(net);

        std::string netInputName;
        for (int i = 0; i < net.node_size(); ++i)
        {
            const tensorflow::NodeDef& node = net.node(i);
            if (node.op() == "Placeholder")
            {
                CV_Assert(i < net.node_size() - 1);
                netInputName = node.name();

                tensorflow::NodeDef* consumer = net.mutable_node(i + 1);
                std::string weights = consumer->input(0);  // Convolution weights
                consumer->clear_input();
                consumer->add_input(netInputName);
                consumer->add_input(weights);
                break;
            }
        }
        CV_Assert(!netInputName.empty());

        // Create SSD postprocessing head.

        addConstNode<int>("concat/axis_flatten", -1, net);
        addConstNode<int>("PriorBox/concat/axis", -2, net);

        for (int i = 0; i < 2; ++i)
        {
            std::string label = i == 0 ? "ClassPredictor" : "BoxEncodingPredictor";
            std::vector<std::string> concatInputs = i == 0 ? classPredictionNodes : boxesPredictionNodes;
            for (int j = 0; j < numLayers; ++j)
            {
                // Flatten predictions
                std::string inpName = concatInputs[j] + "/BiasAdd";
                tensorflow::NodeDef* flatten = addNode("Flatten", inpName + "/Flatten", inpName, net);
                concatInputs[j] = flatten->name();
            }
            addConcatNode(label + "/concat", concatInputs, "concat/axis_flatten", net);
        }

        for (int i = 0; i < numLayers; ++i)
        {
            nodesMapIt = nodesMap.find(boxesPredictionNodes[i] + "/Conv2D");
            CV_Assert(nodesMapIt != nodesMap.end());
            addAttr<bool>("loc_pred_transposed", true, nodesMapIt->second);
        }

        // Add layers that generate anchors (bounding boxes proposals).
        std::vector<float> scales(1 + numLayers, 1.0f);
        for (int i = 0; i < numLayers; ++i)
        {
            scales[i] = minScale + i * (maxScale - minScale) / (numLayers - 1);
        }

        std::vector<std::string> priorBoxesNames;

        for (int i = 0; i < numLayers; ++i)
        {
            tensorflow::NodeDef* priorBox = net.add_node();
            priorBox->set_name(cv::format("PriorBox_%d", i));
            priorBox->set_op("PriorBox");
            priorBox->add_input(boxesPredictionNodes[i] + "/BiasAdd");
            priorBox->add_input(netInputName);

            addAttr<bool>("flip", false, priorBox);
            addAttr<bool>("clip", false, priorBox);

            std::vector<float> widths, heights;
            if (i == 0 && reduceBoxesInLowestLayer)
            {
                widths.push_back(0.1);
                widths.push_back(minScale * sqrt(2.0f));
                widths.push_back(minScale * sqrt(0.5f));

                heights.push_back(0.1);
                heights.push_back(minScale / sqrt(2.0f));
                heights.push_back(minScale / sqrt(0.5f));
            }
            else
            {
                for (int j = 0; j < aspectRatios.size(); ++j)
                {
                    float ar = aspectRatios[j];
                    widths.push_back(scales[i] * sqrt(ar));
                    heights.push_back(scales[i] / sqrt(ar));
                }
                widths.push_back(sqrt(scales[i] * scales[i + 1]));
                heights.push_back(sqrt(scales[i] * scales[i + 1]));
            }
            for (int j = 0; j < widths.size(); ++j)
            {
                widths[j] *= imgWidth;
                heights[j] *= imgHeight;
            }

            addAttr("width", widths, priorBox);
            addAttr("height", heights, priorBox);
            std::vector<float> variance(4);
            variance[0] = variance[1] = 0.1;
            variance[2] = variance[3] = 0.2;
            addAttr("variance", variance, priorBox);
            priorBoxesNames.push_back(priorBox->name());
        }
        addConcatNode("PriorBox/concat", priorBoxesNames, "concat/axis_flatten", net);

        // Sigmoid for classes predictions and DetectionOutput layer
        tensorflow::NodeDef* sigmoid = addNode("Sigmoid", "ClassPredictor/concat/sigmoid",
                                               "ClassPredictor/concat", net);

        addDetectionOutputNode("detection_out", "BoxEncodingPredictor/concat",
                               sigmoid->name(), "PriorBox/concat", numClasses + 1,
                               iouThresh, scoreThresh, 100, net);
        removeUnconnectedNodes(net, "detection_out");
    }

private:
    virtual bool toRemove(const tensorflow::NodeDef& node) CV_OVERRIDE
    {
        const std::string& op = node.op();
        const std::string& name = node.name();
        return keepOps.find(op) == keepOps.end() ||
               name.find("MultipleGridAnchorGenerator") == 0 ||
               name.find("Postprocessor") == 0 ||
               name.find("Preprocessor") == 0;
    }

    std::set<std::string> keepOps;
};

bool simplifyNetFromObjectDetectionAPI(const char* config, tensorflow::GraphDef* net)
{
    std::ifstream ifs(config);
    if (!ifs.is_open())
        CV_Error(Error::StsNotImplemented, cv::format("Failed to open a file %s", config));

    ifs.seekg(0, std::ios::end);
    std::string content((int)ifs.tellg(), ' ');
    ifs.seekg(0, std::ios::beg);
    ifs.read(&content[0], content.size());

    content = "{" + content + "}";
    std::vector<std::string> tokens;
    tokenize(content, tokens);

    // for (int i = 0; i < tokens.size(); ++i)
    // {
    //   std::cout << tokens[i] << '\n';
    // }

    if (tokens.size() > 1 && tokens[1] == "model")
    {
        std::vector<std::string>::iterator tokenIt = tokens.begin();
        TextMessageNode rootMsg(parseTextMessage(tokenIt));

        if (rootMsg["model"].has("ssd"))
        {
            SSDSimplifier simplifier;
            simplifier.process(*net, rootMsg["model"]["ssd"]);
        }
        else if (rootMsg["model"].has("faster_rcnn"))
        {
            FasterRCNNSimplifier simplifier;
            simplifier.process(*net, rootMsg["model"]["faster_rcnn"]);
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported object detection model");
            return false;
        }
        return true;
    }
    else
        return false;
}

CV__DNN_EXPERIMENTAL_NS_END
}}  // namespace dnn, cv

#endif  // HAVE_PROTOBUF
