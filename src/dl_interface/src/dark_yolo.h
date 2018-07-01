#ifndef DARK_YOLO_H
#define DARK_YOLO_H

#define __APP_NAME__ "dark_yolo"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>

#include <rect_class_score.h>

#include <opencv2/opencv.hpp>

extern "C"
{
#undef __cplusplus
#include "box.h"
#include "image.h"
#include "network.h"
#include "detection_layer.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#include "image.h"
#define __cplusplus
}

namespace Yolo
{
    enum YoloClasses//using coco for default cfg and weights
    {
        PERSON, BICYCLE, CAR, MOTORBIKE, AEROPLANE, BUS, TRAIN, TRUCK, BOAT, TRAFFIC_LIGHT,
        FIRE_HYDRANT, STOP_SIGN, PARKING_METER, BENCH, BIRD, CAT, DOG, HORSE, SHEEP, COW,
        ELEPHANT, BEAR, ZEBRA, GIRAFFE, BACKPACK, UMBRELLA, HANDBAG, TIE, SUITCASE, FRISBEE,
        SKIS, SNOWBOARD, SPORTS_BALL, KITE, BASEBALL_BAT, BASEBALL_GLOVE, SKATEBOARD, SURFBOARD, TENNIS_RACKET, BOTTLE,
        WINE_GLASS, CUP, FORK, KNIFE, SPOON, BOWL, BANANA, APPLE, SANDWICH, ORANGE,
        BROCCOLI, CARROT, HOT_DOG, PIZZA, DONUT, CAKE, CHAIR, SOFA, POTTEDPLANT, BED,
        DININGTABLE, TOILET, TVMONITOR, LAPTOP, MOUSE, REMOTE, KEYBOARD, CELL_PHONE, MICROWAVE, OVEN,
        TOASTER, SINK, REFRIGERATOR, BOOK, CLOCK, VASE, SCISSORS, TEDDY_BEAR, HAIR_DRIER, TOOTHBRUSH,
    };
}

namespace darknet {
    class Yolo { //Yolo3Detector
    private:
        double min_confidence_, nms_threshold_;
        network* darknet_network_;
        std::vector<box> darknet_boxes_;
        std::vector<RectClassScore<float> > forward(image &in_darknet_image);
    public:
        Yolo() {}

        void load(std::string &in_model_file, std::string &in_trained_file, double in_min_confidence,
                  double in_nms_threshold);

        ~Yolo();

        image convert_image(const sensor_msgs::ImageConstPtr &in_image_msg);

        std::vector<RectClassScore<float> > detect(image &in_darknet_image);

        uint32_t get_network_width();

        uint32_t get_network_height();


    };
}  // namespace darknet

// class YoloNode {
//     // ros::Subscriber subscriber_image_raw_;
//     ros::Subscriber subscriber_yolo_config_;
//     ros::Publisher publisher_car_objects_;
//     ros::Publisher publisher_person_objects_;
//     ros::NodeHandle node_handle_;
//
//     // darknet::Yolo yolo_detector_;
//
//     image darknet_image_ = {};
//
//     float score_threshold_;
//     float nms_threshold_;
//     double image_ratio_;//resdize ratio used to fit input image to network input size
//     uint32_t image_top_bottom_border_;//black strips added to the input image to maintain aspect ratio while resizing it to fit the network input size
//     uint32_t image_left_right_border_;
//
//     // void convert_rect_to_image_obj(std::vector< RectClassScore<float> >& in_objects, autoware_msgs::image_obj& out_message, std::string in_class);
//     void rgbgr_image(image& im);
//     image convert_ipl_to_image(const sensor_msgs::ImageConstPtr& msg);
//
//     // void config_cb(const autoware_msgs::ConfigSsd::ConstPtr& param);
// public:
//     void Run();
//     void image_callback(const sensor_msgs::ImageConstPtr& in_image_message);
// };

#endif  // DARK_YOLO_H
