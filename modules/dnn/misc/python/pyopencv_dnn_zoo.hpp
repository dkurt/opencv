#ifdef HAVE_OPENCV_DNN

#include <iostream>

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.hpp>

static std::string getSHA(const std::string& path)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    std::string fileOpen = "f = open('" + path + "', 'rb')";
    PyObject* pModule = PyImport_AddModule("__main__");

    PyRun_SimpleString("import hashlib");
    PyRun_SimpleString("sha = hashlib.sha256()");
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("sha.update(f.read())");
    PyRun_SimpleString("f.close()");
    PyRun_SimpleString("sha256 = sha.hexdigest()");

    PyObject* sha256 = PyObject_GetAttrString(pModule, "sha256");
    std::string sha;
    getUnicodeString(sha256, sha);

    Py_DECREF(sha256);
    PyGILState_Release(gstate);
    return sha;
}

static void extractAndRemove(const std::string& archive)
{
    PyGILState_STATE gstate = PyGILState_Ensure();
    std::string cmd = "archive = '" + archive + "'";
    PyRun_SimpleString(cmd.c_str());

    std::string archiveOpen;
    if (archive.size() >= 7 && archive.substr(archive.size() - 7) == ".tar.gz")
    {
        PyRun_SimpleString("import tarfile");
        PyRun_SimpleString("f = tarfile.open(archive)");
    }
    else if (archive.size() >= 4 && archive.substr(archive.size() - 4) == ".zip")
    {
        PyRun_SimpleString("from zipfile import ZipFile");
        PyRun_SimpleString("f = ZipFile(archive, 'r')");
    }
    else
        CV_Error(Error::StsNotImplemented, "Unexpected archive extension: " + archive);

    PyRun_SimpleString("import os");
    PyRun_SimpleString("f.extractall(path=os.path.dirname(archive))");
    PyRun_SimpleString("f.close()");
    // PyRun_SimpleString("os.remove(archive)");

    PyGILState_Release(gstate);
}

static void downloadFile(const std::string& url, const std::string& sha,
                         const std::string& path, bool isArchive = false)
{
    if (utils::fs::exists(path))
    {
        if (!sha.empty())
        {
            std::string currSHA = getSHA(path);
            if (sha != currSHA)
            {
                // We won't download this file because in case of outdated SHA all
                // the applications will download it and still have hash mismatch.
                CV_LOG_WARNING(NULL, "Hash mismatch for " + path + "\n" + "expected: " + sha + "\ngot:      " + currSHA);
            }
        }
        return;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();

    utils::fs::createDirectories(utils::fs::getParent(path));

    std::string urlOpen = "r = urlopen('" + url + "')";
    std::string fileOpen = "f = open('" + path + "', 'wb')";
    std::string printSize = "d = dict(r.info()); size = '<unknown>'\n" \
                            "if 'content-length' in d: size = int(d['content-length']) // MB\n" \
                            "elif 'Content-Length' in d: size = int(d['Content-Length']) // MB\n" \
                            "print('  %s %s [%s MB]' % (r.getcode(), r.msg, size))";

#if PY_MAJOR_VERSION >= 3
    PyRun_SimpleString("from urllib.request import urlopen, Request");
#else
    PyRun_SimpleString("from urllib2 import urlopen, Request");
#endif
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("MB = 1024*1024");
    PyRun_SimpleString("BUFSIZE = 10*MB");

    std::string info = "print('get ' + '" + url + "')";
    PyRun_SimpleString(info.c_str());
    if (PyRun_SimpleString(urlOpen.c_str()) == -1)
    {
        PyGILState_Release(gstate);
        CV_Error(Error::StsError, "Failed to download a file");
    }
    PyRun_SimpleString(printSize.c_str());
    PyRun_SimpleString("import sys; sys.stdout.write('  progress '); sys.stdout.flush()");
    PyRun_SimpleString("buf = r.read(BUFSIZE)");

    if (url.find("drive.google.com") != std::string::npos)
    {
        // For Google Drive we need to add confirmation code for large files
        PyRun_SimpleString("import re");
        PyRun_SimpleString("matches = re.search(b'confirm=(\\w+)&', buf)");

        std::string cmd = "if matches: " \
                          "cookie = r.headers.get('Set-Cookie'); " \
                          "req = Request('" + url + "' + '&confirm=' + matches.group(1).decode('utf-8')); " \
                          "req.add_header('cookie', cookie); " \
                          "r = urlopen(req); " \
                          "buf = r.read(BUFSIZE)";  // Reread first chunk
        PyRun_SimpleString(cmd.c_str());
    }
    PyRun_SimpleString("while buf: sys.stdout.write('>'); sys.stdout.flush(); " \
                       "f.write(buf); buf = r.read(BUFSIZE)");
    PyRun_SimpleString("sys.stdout.write('\\n'); f.close()");
    PyGILState_Release(gstate);

    if (!sha.empty())
    {
        std::string resSHA = getSHA(path);
        if (sha != resSHA)
            CV_LOG_WARNING(NULL, "Hash mismatch for " + path + "\n" + "expected: " + sha + "\ngot:      " + resSHA);
    }

    if (isArchive)
        extractAndRemove(path);
}

static void getFileInfo(const FileNode& entry, std::string& name,
                        std::string& url, std::string& sha)
{
    name = entry["name"].string();
    url = entry["source"].string();
    sha = entry["sha256"].string();
}

static void yamlToJSON(const std::string& yaml, const std::string& json)
{
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyRun_SimpleString("import yaml");
    std::string fileOpen = "f = open('" + yaml + "', 'r')";
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("data = yaml.safe_load(f)");
    PyRun_SimpleString("f.close()");

    PyRun_SimpleString("import json");
    fileOpen = "f = open('" + json + "', 'wt')";
    PyRun_SimpleString(fileOpen.c_str());
    PyRun_SimpleString("json.dump(data, f)");
    PyRun_SimpleString("f.close()");

    PyGILState_Release(gstate);
}

// static void resolveFiles(const FileStorage& fs,
//                          std::string& modelPath, std::string& modelURL, std::string& modelSHA,
//                          std::string& configPath, std::string& configURL, std::string& configSHA)
// {
//     const FileNode& files = fs["files"];
//
//
// }

static std::vector<std::string> zooNames;
static std::vector<std::string> zooConfigs;
static std::vector<PyMethodDef> zooMethods;

static PyObject* pyopencv_cv_dnn_zoo_topology(PyObject* self, PyObject* args, PyObject* kw)
{
    std::string name;
    getUnicodeString(self, name);
    std::cout << "process: " << name << '\n';

    // Open a config file.
    int idx = std::find(zooNames.begin(), zooNames.end(), name) - zooNames.begin();
    std::cout << "config: " << zooConfigs[idx] << '\n';

    yamlToJSON(zooConfigs[idx], zooConfigs[idx] + ".json");

    FileStorage fs(zooConfigs[idx] + ".json", FileStorage::READ);

    std::cout << fs["framework"].string() << '\n';

    // Determine weights and optional text file.
    std::string modelPath, modelURL, modelSHA;
    getFileInfo(fs["files"][0], modelPath, modelURL, modelSHA);

    std::cout << modelPath << std::endl;
    std::cout << modelURL << std::endl;
    std::cout << modelSHA << std::endl;


    dnn::zoo::Topology t("", "");

    return pyopencv_from(t);
}

static void initDnnZoo(PyObject* m)
{
    PyObject* d = PyModule_GetDict(m);
    printf("EHLO!\n");

    // Download models configs.
    // std::string cache = utils::fs::getCacheDirectory("open_model_zoo_cache", "OPENCV_OPEN_MODEL_ZOO_CACHE_DIR");
    // extractAndRemove("models/open_model_zoo.zip");
    // downloadFile("https://github.com/opencv/open_model_zoo/archive/master.zip",
    //              "", utils::fs::join("models", "open_model_zoo.zip"), true);

    // Register the models.
    // std::string prefixes[] = {"intel", }
    utils::fs::glob("models/open_model_zoo-master/models", "*.yml", zooConfigs, true);

    zooNames.resize(zooConfigs.size());
    zooMethods.resize(zooConfigs.size());
    for (size_t i = 0; i < zooConfigs.size(); ++i)
    {
        std::string name = utils::fs::getParent(zooConfigs[i]);
        name = name.substr(utils::fs::getParent(name).size() + 1);
        std::replace(name.begin(), name.end(), '-', '_');
        std::replace(name.begin(), name.end(), '.', '_');

        zooNames[i] = name;
        zooMethods[i] = {
            zooNames[i].c_str(),
            CV_PY_FN_WITH_KW_(pyopencv_cv_dnn_zoo_topology, METH_STATIC)
        };

        PyObject* funcName = PyString_FromString(zooMethods[i].ml_name);
        PyObject* func = PyCFunction_NewEx(&zooMethods[i], funcName, NULL);
        PyDict_SetItemString(d, zooMethods[i].ml_name, func);
        Py_DECREF(func);
        Py_DECREF(funcName);
    }
}

#endif  // HAVE_OPENCV_DNN
