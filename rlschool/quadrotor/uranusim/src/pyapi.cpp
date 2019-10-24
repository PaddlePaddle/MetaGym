#include <Python.h>
#include <structmember.h>
#include <string>
#include "uranus_simulator.h"

static PyObject *s_uranusim_err;

/**********************************
 *                                *
 *            OBJECT              *
 *           URANUSIM             *
 *                                *
 *                                *
 **********************************/

/*uranusim Object*/
typedef struct{
    PyObject_HEAD;
    uranus::uranusim::Simulator *DATA;
} py_uranusim;

/*uranus Simulator, perform one step of update
 *Inputs: U1~U4
 *Output: None
 */
static PyObject* py_uranusim_step(py_uranusim *self, PyObject* Argvs){
    PyObject *u = NULL;
    float dt = 0;

    if (!(self->DATA)){
        PyErr_SetString(s_uranusim_err, "unexpected error, NULL data");
        return NULL;
    }

    if (!PyArg_ParseTuple(Argvs, "Of", &u, &dt)){
        PyErr_SetString(s_uranusim_err,
                "function require list of commands as input");
        return NULL;
    }
    if (!PyList_Check(u)){
        PyErr_SetString(s_uranusim_err,
                "expect input as a list");
        return NULL;
    }
    if (PyList_GET_SIZE(u) != 4){
        PyErr_SetString(s_uranusim_err,
                "expect input command as a list of length 4");
        return NULL;
    }
    float u1 = PyFloat_AsDouble(PyList_GetItem(u, 0));
    float u2 = PyFloat_AsDouble(PyList_GetItem(u, 1));
    float u3 = PyFloat_AsDouble(PyList_GetItem(u, 2));
    float u4 = PyFloat_AsDouble(PyList_GetItem(u, 3));

    uranus::uranusim::ActionU act(u1, u2, u3, u4);
    int ret = (self->DATA)->run_time(act, dt);
    // std::cout << "ret" << ret << std::endl;

    if (ret != 0){
        PyErr_SetString(s_uranusim_err,
                "Unexpected Error In simulation, Check if correctly configured");
        return NULL;
    }

    Py_RETURN_NONE;
}

/*  Python Class Internal Constructor */
static PyObject *py_uranusim_new(
        PyTypeObject *type, PyObject *args, PyObject *kwds){
    py_uranusim *self;
    self = (py_uranusim *)type->tp_alloc(type, 0);
    return (PyObject *)self;
}

/* Get the current state of simulator*/
static PyObject *py_uranusim_getstate(
        py_uranusim* Self, PyObject *args, PyObject *kwds){

    if (!(Self->DATA)){
        PyErr_SetString(s_uranusim_err, "unexpected error, NULL data");
        return NULL;
    }

    PyObject *ret_dict = PyDict_New();
    if (!ret_dict){
        PyErr_SetString(s_uranusim_err,
                "unexpected error getting states");
        return NULL;
    }
    Py_INCREF(ret_dict);

    uranus::uranusim::PrintableState state;
    Self->DATA->get_printable_state(state);

    PyDict_SetItemString(ret_dict, "roll",
            PyFloat_FromDouble(state.roll));
    PyDict_SetItemString(ret_dict, "pitch",
            PyFloat_FromDouble(state.pitch));
    PyDict_SetItemString(ret_dict, "yaw",
            PyFloat_FromDouble(state.yaw));
    PyDict_SetItemString(ret_dict, "x",
            PyFloat_FromDouble(state.g_x));
    PyDict_SetItemString(ret_dict, "y",
            PyFloat_FromDouble(state.g_y));
    PyDict_SetItemString(ret_dict, "z",
            PyFloat_FromDouble(state.g_z));
    PyDict_SetItemString(ret_dict, "w_x",
            PyFloat_FromDouble(state.w_x));
    PyDict_SetItemString(ret_dict, "w_y",
            PyFloat_FromDouble(state.w_y));
    PyDict_SetItemString(ret_dict, "w_z",
            PyFloat_FromDouble(state.w_z));
    PyDict_SetItemString(ret_dict, "g_v_x",
            PyFloat_FromDouble(state.g_v_x));
    PyDict_SetItemString(ret_dict, "g_v_y",
            PyFloat_FromDouble(state.g_v_y));
    PyDict_SetItemString(ret_dict, "g_v_z",
            PyFloat_FromDouble(state.g_v_z));
    PyDict_SetItemString(ret_dict, "b_v_x",
            PyFloat_FromDouble(state.b_v_x));
    PyDict_SetItemString(ret_dict, "b_v_y",
            PyFloat_FromDouble(state.b_v_y));
    PyDict_SetItemString(ret_dict, "b_v_z",
            PyFloat_FromDouble(state.b_v_z));

    return ret_dict;
}

/* Get the current sensor output of simulator*/
static PyObject *py_uranusim_getsensor(
        py_uranusim* Self, PyObject *args, PyObject *kwds){

    if (!(Self->DATA)){
        PyErr_SetString(s_uranusim_err, "unexpected error, NULL data");
        return NULL;
    }

    PyObject *ret_dict = PyDict_New();
    if (!ret_dict){
        PyErr_SetString(s_uranusim_err,
                "unexpected error getting states");
        return NULL;
    }
    Py_INCREF(ret_dict);

    uranus::uranusim::SensorOutput output;
    Self->DATA->read_sensor(output);

    PyDict_SetItemString(ret_dict, "imu_x",
            PyFloat_FromDouble(output.imu_acc(0)));
    PyDict_SetItemString(ret_dict, "imu_y",
            PyFloat_FromDouble(output.imu_acc(1)));
    PyDict_SetItemString(ret_dict, "imu_z",
            PyFloat_FromDouble(output.imu_acc(2)));
    PyDict_SetItemString(ret_dict, "gyro_x",
            PyFloat_FromDouble(output.imu_gyro(0)));
    PyDict_SetItemString(ret_dict, "gyro_y",
            PyFloat_FromDouble(output.imu_gyro(1)));
    PyDict_SetItemString(ret_dict, "gyro_z",
            PyFloat_FromDouble(output.imu_gyro(2)));
    PyDict_SetItemString(ret_dict, "vio_x",
            PyFloat_FromDouble(output.vio(0)));
    PyDict_SetItemString(ret_dict, "vio_y",
            PyFloat_FromDouble(output.vio(1)));
    PyDict_SetItemString(ret_dict, "vio_z",
            PyFloat_FromDouble(output.vio(2)));

    return ret_dict;
}

/* Read configuration */
static PyObject *py_uranusim_getconfig(
        py_uranusim* Self, PyObject *args, PyObject *kwds){
    char *filename = NULL;
    if (!PyArg_ParseTuple(args, "s", &filename)){
        PyErr_SetString(s_uranusim_err,
                "function require a file name as input");
        return NULL;
    }
    if (Self->DATA->get_config(filename) != 0){
        PyErr_SetString(s_uranusim_err,
                "Unexpected Error Getting Configurations");
        return NULL;
    }
    Py_RETURN_NONE;
}


/* Reset the current state of simulator*/
static PyObject *py_uranusim_reset(
        py_uranusim* Self, PyObject *args, PyObject *kwds){

    static char *kwlist[] = {"roll", "pitch", "yaw",
            "w_x", "w_y", "w_z",
            "g_v_x", "g_v_y", "g_v_z",
            "x", "y", "z", NULL};

    float roll = 0.0;
    float pitch = 0.0;
    float yaw = 0.0;
    float w_x = 0.0;
    float w_y = 0.0;
    float w_z = 0.0;
    float g_v_x = 0.0;
    float g_v_y = 0.0;
    float g_v_z = 0.0;
    float x = 0.0;
    float y = 0.0;
    float z = 0.0;

    if (!(Self->DATA)){
        PyErr_SetString(s_uranusim_err, "unexpected error, NULL data");
        return NULL;
    }

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|fffffffff", kwlist,
            &roll, &pitch, &yaw, &w_x, &w_y, &w_z,
            &g_v_x, &g_v_y, &g_v_z, &x, &y, &z)){
        PyErr_SetString(s_uranusim_err, "unexpected error in parsing args");
        return NULL;
    }

    Self->DATA->reset(
            x, y, z,
            g_v_x, g_v_y, g_v_z,
            w_x, w_y, w_z,
            roll, pitch, yaw
            );

    Py_RETURN_NONE;
}

/*  Python Class Initializer, same as "__init__" */
static int py_uranusim_init(
        py_uranusim* Self, PyObject *args, PyObject *kwds){

    Self->DATA = new uranus::uranusim::Simulator();
    return 0;
}


/*  Python Class Destructor*/
static void py_uranusim_destruct(py_uranusim* Self){
    if (!(Self->DATA)){
        delete Self->DATA;
    }
    Self->ob_type->tp_free((PyObject*)Self);
}

/* "__str__" in python class */
static PyObject* py_uranusim_str(py_uranusim* Self){
    if (!(Self->DATA)){
        PyErr_SetString(s_uranusim_err, "unexpected error, NULL data");
        return NULL;
    }
    uranus::uranusim::PrintableState state;
    Self->DATA->get_printable_state(state);
    std::string output = "uranusim Object:";
    output += "Configuration:{";
    output += "State:{";
    output += "x:" + std::to_string(state.g_x);
    output += ", y:" + std::to_string(state.g_y);
    output += ", z:" + std::to_string(state.g_z);
    output += ", roll:" + std::to_string(state.roll);
    output += ", pitch:" + std::to_string(state.pitch);
    output += ", yaw:" + std::to_string(state.yaw);
    output += ", g_v_x:" + std::to_string(state.g_v_x);
    output += ", g_v_y:" + std::to_string(state.g_v_y);
    output += ", g_v_z:" + std::to_string(state.g_v_z);
    output += ", w_x:" + std::to_string(state.w_x);
    output += ", w_y:" + std::to_string(state.w_y);
    output += ", w_z:" + std::to_string(state.w_z);
    output += ", b_v_x:" + std::to_string(state.b_v_x);
    output += ", b_v_y:" + std::to_string(state.b_v_y);
    output += ", b_v_z:" + std::to_string(state.b_v_z);
    output += "}}";
    return Py_BuildValue("s", output.c_str());
}

static PyObject* py_uranusim_repr(py_uranusim* Self) {
    return py_uranusim_str(Self);
}


/*   类的所有成员函数结构列表   */

static PyMethodDef s_py_uranusim_method_members[] = {
    {"Step", (PyCFunction)py_uranusim_step, METH_VARARGS,
        "Perform a step of simulation, need a command list of length 4"},
    {"Reset", (PyCFunction)py_uranusim_reset, METH_VARARGS | METH_KEYWORDS,
        "Reset the state of the simulator"},
    {"GetState", (PyCFunction)py_uranusim_getstate, METH_VARARGS,
        "Get the dictionary state of the simulator"},
    {"GetSensor", (PyCFunction)py_uranusim_getsensor, METH_VARARGS,
        "Get the sensor output of the UAV"},
    {"GetConfig", (PyCFunction)py_uranusim_getconfig, METH_VARARGS,
        "Read UAV Configuration from predefined XML"},
    {NULL}
};

/* 类成员 */
static PyMemberDef s_py_uranusim_data_members[] = {
        {"DATA",   T_STRING, offsetof(py_uranusim, DATA),   0, "Storage Class"},
        {NULL}
};

/* 类申明 */
static PyTypeObject s_py_uranusim_classinfo = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "uranusim",               /*tp_name*/
    sizeof(py_uranusim),    /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)py_uranusim_destruct, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)py_uranusim_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    (reprfunc)py_uranusim_str,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT, /*tp_flags*/
    "uranus Simulator Class, initialize with parameters (dt)",  /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    s_py_uranusim_method_members,             /* tp_methods */
    s_py_uranusim_data_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)py_uranusim_init,      /* tp_init */
    0,                         /* tp_alloc */
    py_uranusim_new                 /* tp_new */
};

static PyMethodDef s_uranusim_methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

/* Module Initialization */
PyMODINIT_FUNC
initliburanusim(void) {
    PyObject* p_return = NULL;
    if (PyType_Ready(&s_py_uranusim_classinfo) < 0){
        PyErr_PrintEx(0);
        return;
    }

    s_uranusim_err = PyErr_NewException("liburanusim.uranusimulationError",
            NULL, NULL);
    Py_INCREF(s_uranusim_err);

    p_return = Py_InitModule3("liburanusim", s_uranusim_methods,
       "An uranus Simulator");
    if (p_return == NULL){
        return;
    }

    // add new object to module
    PyModule_AddObject(p_return, "uranusim", (PyObject*)&s_py_uranusim_classinfo);
    Py_INCREF(&s_py_uranusim_classinfo);
}
