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
        else if (symbol == '{' || symbol == '}')
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
    for (;;)
    {
        ++tokenIt;
        fieldName = *tokenIt;

        if (fieldName == "}")
            break;

        ++tokenIt;
        fieldValue = *tokenIt;

        if (fieldValue == "{")
            msg->messages.insert(std::make_pair(fieldName, parseTextMessage(tokenIt)));
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

static tensorflow::NodeDef* addNode(const std::string& type, const std::string& name,
                                    const std::string& input, tensorflow::GraphDef& net)
{
    tensorflow::NodeDef* node = net.add_node();
    node->set_name(name);
    node->set_op(type);
    node->add_input(input);
    return node;
}

static void simplifySSD(tensorflow::GraphDef& net, const TextMessageNode& config)
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

    // Remove extra nodes and attributes.
    std::vector<std::string> removedNodes, keepOps, unusedAttrs;

    keepOps.push_back("Conv2D");
    keepOps.push_back("BiasAdd");
    keepOps.push_back("Add");
    keepOps.push_back("Relu6");
    keepOps.push_back("Placeholder");
    keepOps.push_back("FusedBatchNorm");
    keepOps.push_back("DepthwiseConv2dNative");
    keepOps.push_back("ConcatV2");
    keepOps.push_back("Mul");
    keepOps.push_back("MaxPool");
    keepOps.push_back("AvgPool");
    keepOps.push_back("Identity");
    keepOps.push_back("Const");

    unusedAttrs.push_back("T");
    unusedAttrs.push_back("data_format");
    unusedAttrs.push_back("Tshape");
    unusedAttrs.push_back("N");
    unusedAttrs.push_back("Tidx");
    unusedAttrs.push_back("Tdim");
    unusedAttrs.push_back("use_cudnn_on_gpu");
    unusedAttrs.push_back("Index");
    unusedAttrs.push_back("Tperm");
    unusedAttrs.push_back("is_training");
    unusedAttrs.push_back("Tpaddings");

    // Remove unused nodes and attributes.
    RepeatedPtrField<tensorflow::NodeDef>::iterator it;
    google::protobuf::Map<std::string, tensorflow::AttrValue>::iterator attrIt;
    for (it = net.mutable_node()->begin(); it != net.mutable_node()->end();)
    {
        tensorflow::NodeDef& layer = *it;
        std::string op = layer.op();
        std::string name = layer.name();

        if (std::find(keepOps.begin(), keepOps.end(), op) == keepOps.end() ||
            name.find("MultipleGridAnchorGenerator") == 0 ||
            name.find("Postprocessor") == 0 ||
            name.find("Preprocessor") == 0)
        {
            removedNodes.push_back(name);
            it = net.mutable_node()->erase(it);
        }
        else
            ++it;

        for (attrIt = layer.mutable_attr()->begin(); attrIt != layer.mutable_attr()->end();)
        {
            if (std::find(unusedAttrs.begin(), unusedAttrs.end(), attrIt->first) != unusedAttrs.end())
                attrIt = layer.mutable_attr()->erase(attrIt);
            else
                ++attrIt;
        }
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

    tensorflow::NodeDef* detectionOut = net.add_node();
    detectionOut->set_name("detection_out");
    detectionOut->set_op("DetectionOutput");

    detectionOut->add_input("BoxEncodingPredictor/concat");
    detectionOut->add_input(sigmoid->name());
    detectionOut->add_input("PriorBox/concat");

    addAttr<int>("num_classes", numClasses + 1, detectionOut);
    addAttr<bool>("share_location", true, detectionOut);
    addAttr<int>("background_label_id", 0, detectionOut);
    addAttr<float>("nms_threshold", iouThresh, detectionOut);
    addAttr<int>("top_k", 100, detectionOut);
    addAttr<std::string>("code_type", "CENTER_SIZE", detectionOut);
    addAttr<int>("keep_top_k", 100, detectionOut);
    addAttr<float>("confidence_threshold", scoreThresh, detectionOut);

    std::vector<std::string> unconnected;
    for (;;)
    {
        getUnconnectedNodes(net, unconnected);
        unconnected.erase(std::remove(unconnected.begin(), unconnected.end(), detectionOut->name()),
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

    if (tokens.size() > 1 && tokens[1] == "model")
    {
        std::vector<std::string>::iterator tokenIt = tokens.begin();
        TextMessageNode rootMsg(parseTextMessage(tokenIt));

        if (rootMsg["model"].has("ssd"))
            simplifySSD(*net, rootMsg["model"]["ssd"]);
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
