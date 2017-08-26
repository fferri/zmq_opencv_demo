#include <iostream>
#include <string>
#include <deque>
#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include "zhelpers.hpp"
#include "opencv_conversions.hpp"

// camera thread, a.k.a. the ventilator
//
// this will try to sample camera frames evenly, according to average
// according to the workers' average processing time (to reduce jitter)

void camera(zmq::context_t &ctx, cv::VideoCapture &cap, cv::Size sz, int num_workers)
{
    zmq::socket_t sender(ctx, ZMQ_PUSH);
    sender.bind("inproc://cam");
    zmq::socket_t receiver(ctx, ZMQ_PULL); // feedback about processing time from workers
    receiver.bind("inproc://proc_time");

    size_t i = 0;
    std::vector<double> tv;
    for(int j = 0; j < 8; j++) tv.push_back(0.0);
    double average_processing_time = 0.0;

    cv::Mat frame;
    while(true)
    {
        cap >> frame;
        cv::resize(frame, frame, sz);
        zmq::message_t msg;
        cv::to_zmq(frame, msg);
        sender.send(msg);

        // receive processing time from workers, compute average, and adjust delay:
        while(true)
        {
            zmq::pollitem_t poll_items[] = {{receiver, 0, ZMQ_POLLIN, 0}};
            zmq::poll(&poll_items[0], 1, 0);
            if(poll_items[0].revents & ZMQ_POLLIN)
            {
                zmq::message_t msg;
                receiver.recv(&msg);
                tv[i++] = *(static_cast<double*>(msg.data()));
                if(i >= tv.size()) i = 0;
            }
            else break;
        }
        average_processing_time = 0.0;
        for(int i = 0; i < tv.size(); i++)
            average_processing_time += tv[i];
        average_processing_time /= tv.size();

        boost::this_thread::sleep(boost::posix_time::milliseconds(average_processing_time * 1000 / num_workers));
    }
}

// the worker
//
// will feed back processing time to the ventilator

void worker(zmq::context_t &ctx, int id)
{
    zmq::socket_t receiver(ctx, ZMQ_PULL);
    receiver.setsockopt(ZMQ_CONFLATE, 1); // don't fill queue with messages
    receiver.connect("inproc://cam");
    zmq::socket_t sender(ctx, ZMQ_PUSH);
    sender.connect("inproc://processed");
    zmq::socket_t processing_time(ctx, ZMQ_PUSH); // feedback about processing time
    processing_time.connect("inproc://proc_time");

    zmq::message_t msg;
    cv::Mat img;
    while(true)
    {
        receiver.recv(&msg);
        int64 start = cv::getTickCount();
        cv::from_zmq(img, msg);
        //cv::fastNlMeansDenoisingColored(img, img, 4, 10);
        cv::cvtColor(img, img, CV_BGR2GRAY);
        cv::GaussianBlur(img, img, cv::Size(7, 7), 1.5, 1.5);
        cv::Canny(img, img, 0, 30, 3);
        int64 duration = cv::getTickCount() - start;
        double duration_s = duration / cv::getTickFrequency();
        std::cout << "[worker-" << id << "] " << duration_s << "s" << std::endl;
        cv::to_zmq(img, msg);
        sender.send(msg);

        zmq::message_t msg_time(sizeof(double));
        memcpy(msg_time.data(), &duration_s, sizeof(double));
        processing_time.send(msg_time);
    }
}

int main(int argc, char** argv)
{
    zmq::context_t context(1);

    cv::VideoCapture cap(0);
    if(!cap.isOpened()) return 1;
    cv::namedWindow("img", 1);

    const int num_workers = 4;

    zmq::socket_t receiver(context, ZMQ_PULL);
    receiver.bind("inproc://processed");

    boost::thread camera_thread(&camera, boost::ref(context), cap, cv::Size(640, 360), num_workers);
    for(int i = 0; i < num_workers; i++)
        boost::thread worker_thread(&worker, boost::ref(context), i);

    zmq::message_t msg;
    cv::Mat img;
    int64 start = cv::getTickCount(), frames = 0;
    while(true)
    {
        receiver.recv(&msg);
        cv::from_zmq(img, msg);

        frames++;
        double fps = frames / ((cv::getTickCount() - start) / cv::getTickFrequency());
        std::cout << fps << " FPS" << std::endl;

        cv::imshow("img", img);
        if(cv::waitKey(30) >= 0) break;
    }
    return 0;
}

