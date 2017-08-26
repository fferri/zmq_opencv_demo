#include <opencv2/core/core.hpp>

namespace cv {

static bool
from_zmq(cv::Mat &mat, zmq::message_t &msg)
{
    int *i = (int*)msg.data();
    cv::Mat tmp(i[0], i[1], i[2], (void*)&i[3]);
    tmp.copyTo(mat);
    return true;
}

static bool
to_zmq(const cv::Mat &mat, zmq::message_t &msg)
{
    size_t data_size = mat.total() * mat.elemSize();
    msg.rebuild(data_size + sizeof(int) * 3);
    int *i = (int*)msg.data();
    i[0] = mat.rows;
    i[1] = mat.cols;
    i[2] = mat.type();
    if(mat.isContinuous())
    {
        memcpy((char*)&i[3], mat.data, data_size);
    }
    else
    {
        const size_t rowsize = mat.cols * mat.elemSize();
        char *data = (char*)&i[3];
        for(int i = 0; i < mat.rows; i++)
            memcpy(&data[i * rowsize], mat.ptr(i), rowsize);
    }
    return true;
}

} // namespace cv

