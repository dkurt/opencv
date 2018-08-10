import tensorflow as tf
from tensorflow.core.framework.node_def_pb2 import NodeDef
from google.protobuf import text_format

def tensorMsg(values):
    if all([isinstance(v, float) for v in values]):
        dtype = 'DT_FLOAT'
        field = 'float_val'
    elif all([isinstance(v, int) for v in values]):
        dtype = 'DT_INT32'
        field = 'int_val'
    else:
        raise Exception('Wrong values types')

    msg = 'tensor { dtype: ' + dtype + ' tensor_shape { dim { size: %d } }' % len(values)
    for value in values:
        msg += '%s: %s ' % (field, str(value))
    return msg + '}'

def addConstNode(name, values, graph_def):
    node = NodeDef()
    node.name = name
    node.op = 'Const'
    text_format.Merge(tensorMsg(values), node.attr["value"])
    graph_def.node.extend([node])


def tokenize(s):
    tokens = []
    token = ""
    isString = False
    isComment = False
    for symbol in s:
        isComment = (isComment and symbol != '\n') or (not isString and symbol == '#')
        if isComment:
            continue

        if symbol == ' ' or symbol == '\t' or symbol == '\r' or symbol == '\'' or \
           symbol == '\n' or symbol == ':' or symbol == '\"' or symbol == ';' or \
           symbol == ',':

            if token != "" or ((symbol == '\"' or symbol == '\'') and isString):
                tokens.append(token)
                token = ""
            isString = (symbol == '\"' or symbol == '\'') ^ isString;

        elif symbol == '{' or symbol == '}' or symbol == '[' or symbol == ']':
            if token:
                tokens.append(token)
                token = ""
            tokens.append(symbol)
        else:
            token += symbol
    if token:
        tokens.append(token)
    return tokens


def parseTextMessage(tokens, idx):
    msg = {}
    assert(tokens[idx] == '{')

    isArray = False
    while True:
        if not isArray:
            idx += 1
            fieldName = tokens[idx]
            if fieldName == '}':
                break

        idx += 1
        fieldValue = tokens[idx]

        if fieldValue == '{':
            msg[fieldName] = parseTextMessage(tokens, idx)
        elif fieldValue == '[':
            isArray = True
        elif fieldValue == ']':
            isArray = False
        else:
            if not fieldName in msg:
                msg[fieldName] = [fieldValue]
            else:
                msg[fieldName].append(fieldValue)
    return msg


def readTextMessage(filePath):
    with open(filePath, 'rt') as f:
        content = f.read()
        tokens = tokenize('{' + content + '}')
        return parseTextMessage(tokens, 0)


class NodeDef:
    def __init__(self):
        self.input = []
        self.name = ""
        self.op = ""
        self.attr = {}


class GraphDef:
    def __init__(self):
        self.node = []


    # Ptr<TextMessage> msg(new TextMessage());
    # CV_Assert(*tokenIt == "{");
    #
    # std::string fieldName, fieldValue;
    # bool isArray = false;
    # for (;;)
    # {
    #     if (!isArray)
    #     {
    #         ++tokenIt;
    #         fieldName = *tokenIt;
    #
    #         if (fieldName == "}")
    #             break;
    #     }
    #
    #     ++tokenIt;
    #     fieldValue = *tokenIt;
    #
    #     if (fieldValue == "{")
    #         msg->messages.insert(std::make_pair(fieldName, parseTextMessage(tokenIt)));
    #     else if (fieldValue == "[")
    #         isArray = true;
    #     else if (fieldValue == "]")
    #         isArray = false;
    #     else
    #     {
    #         std::map<std::string, std::vector<std::string> >::iterator it = msg->fields.find(fieldName);
    #         if (it == msg->fields.end())
    #             msg->fields.insert(std::make_pair(fieldName, std::vector<std::string>(1, fieldValue)));
    #         else
    #             it->second.push_back(fieldValue);
    #     }
    # }
    # return msg;
