#ifdef HAVE_OPENCV_DNN

#include <iostream>

static PyObject* pyopencv_cv_dnn_zoo_topology(PyObject* self, PyObject* args, PyObject* kw)
{
    // PyGILState_STATE gstate = PyGILState_Ensure();
    //
    // Determine current topology name.
    // PyRun_SimpleString("import inspect");
    // PyRun_SimpleString("name = inspect.stack()[0][3]");
    //
    // PyObject* topologyName_ = PyObject_GetAttrString(PyImport_AddModule("__main__"), "name");
    // std::string topologyName;
    // getUnicodeString(topologyName_, topologyName);
    //
    // std::cout << topologyName << '\n';
    //
    // Py_DECREF(topologyName_);

    std::cout << self << '\n';

    std::cout << PyString_AsString(self) << '\n';
    // std::cout << self << '\n';
    // std::cout << PyObject_Type(self) << '\n';

    dnn::zoo::Topology t("", "");

    // PyGILState_Release(gstate);

    return pyopencv_from(t);
}

static PyMethodDef methods[] = {
  {"example_topology", CV_PY_FN_WITH_KW_(pyopencv_cv_dnn_zoo_topology, METH_STATIC), "redirectError(onError) -> None"},
  {NULL, NULL},
};

static void initDnnZoo(PyObject* m)
{
    PyObject* d = PyModule_GetDict(m);
    printf("EHLO!\n");

    for (PyMethodDef * m = methods; m->ml_name != NULL; ++m)
    {
        // Methods will determine their names by a fake string constant.
        PyObject* name = PyString_FromString(m->ml_name);
        PyObject* method_obj = PyCFunction_NewEx(m, name, NULL);
        PyDict_SetItemString(d, m->ml_name, method_obj);
        Py_DECREF(method_obj);
        Py_DECREF(name);
    }

    // PyObject *key, *value;
    // Py_ssize_t pos = 0;
    //
    // while (PyDict_Next(d, &pos, &key, &value)) {
    //   printf("key %s\n", PyString_AsString(key));
    // }
}

#endif  // HAVE_OPENCV_DNN
