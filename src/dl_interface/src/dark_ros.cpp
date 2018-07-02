#include <ros/ros.h>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <boost/bind.hpp>
#include <dark_msgs/Detect.h>
#include <dark_msgs/DetectArray.h>

#include "dark_yolo.h"

darknet::Yolo yolo_detector_;
darknet::Classify classifier_;

ros::Publisher detect_publish;

namespace darknet
{
    uint32_t Yolo::get_network_height()
    {
        return darknet_network_->h;
    }
    uint32_t Yolo::get_network_width()
    {
        return darknet_network_->w;
    }
    void Yolo::load(std::string& in_model_file, std::string& in_trained_file, double in_min_confidence, double in_nms_threshold)
    {
        min_confidence_ = in_min_confidence;
        nms_threshold_ = in_nms_threshold;
        darknet_network_ = parse_network_cfg(&in_model_file[0]);
        load_weights(darknet_network_, &in_trained_file[0]);
        set_batch_network(darknet_network_, 1);

        layer output_layer = darknet_network_->layers[darknet_network_->n - 1];
        darknet_boxes_.resize(output_layer.w * output_layer.h * output_layer.n);
    }

    Yolo::~Yolo()
    {
        free_network(darknet_network_);
    }

    std::vector< RectClassScore<float> > Yolo::detect(image& in_darknet_image)
    {
        return forward(in_darknet_image);
    }

    image Yolo::convert_image(const sensor_msgs::ImageConstPtr& msg)
    {
        if (msg->encoding != sensor_msgs::image_encodings::BGR8)
        {
            ROS_ERROR("Unsupported encoding");
            exit(-1);
        }

        auto data = msg->data;
        uint32_t height = msg->height, width = msg->width, offset = msg->step - 3 * width;
        uint32_t i = 0, j = 0;
        image im = make_image(width, height, 3);

        for (uint32_t line = height; line; line--)
        {
            for (uint32_t column = width; column; column--)
            {
                for (uint32_t channel = 0; channel < 3; channel++)
                    im.data[i + width * height * channel] = data[j++] / 255.;
                i++;
            }
            j += offset;
        }

        if (darknet_network_->w == (int) width && darknet_network_->h == (int) height)
        {
            return im;
        }
        image resized = resize_image(im, darknet_network_->w, darknet_network_->h);
        free_image(im);
        return resized;
    }

    std::vector< RectClassScore<float> > Yolo::forward(image& in_darknet_image)
    {
        float * in_data = in_darknet_image.data;
        float *prediction = network_predict(darknet_network_, in_data);
        layer output_layer = darknet_network_->layers[darknet_network_->n - 1];

        output_layer.output = prediction;
        int nboxes = 0;
        int num_classes = output_layer.classes;
        detection *darknet_detections = get_network_boxes(darknet_network_, darknet_network_->w, darknet_network_->h, min_confidence_, .5, NULL, 0, &nboxes);

        do_nms_sort(darknet_detections, nboxes, num_classes, nms_threshold_);

        std::vector< RectClassScore<float> > detections;

        for (int i = 0; i < nboxes; i++)
        {
            int class_id = -1;
            float score = 0.f;
            //find the class
            for(int j = 0; j < num_classes; ++j){
                if (darknet_detections[i].prob[j] >= min_confidence_){
                    if (class_id < 0) {
                        class_id = j;
                        score = darknet_detections[i].prob[j];
                    }
                }
            }
            //if class found
            if (class_id >= 0)
            {
                RectClassScore<float> detection;

                detection.x = darknet_detections[i].bbox.x - darknet_detections[i].bbox.w/2;
                detection.y = darknet_detections[i].bbox.y - darknet_detections[i].bbox.h/2;
                detection.w = darknet_detections[i].bbox.w;
                detection.h = darknet_detections[i].bbox.h;
                detection.score = score;
                detection.class_type = class_id;
                //std::cout << detection.toString() << std::endl;

                detections.push_back(detection);
            }
        }
        //std::cout << std::endl;
        return detections;
    }

    uint32_t Classify::get_network_height()
    {
        return classify_network_->h;
    }
    uint32_t Classify::get_network_width()
    {
        return classify_network_->w;
    }
    void Classify::load(std::string& in_model_file, std::string& in_trained_file)
    {
        classify_network_ = parse_network_cfg(&in_model_file[0]);
        // load_weights(classify_network_, &in_trained_file[0]);
        // set_batch_network(classify_network_, 1);


        // layer output_layer = classify_network_->layers[classify_network_->n - 1];
        // darknet_boxes_.resize(output_layer.w * output_layer.h * output_layer.n);
    }

    Classify::~Classify()
    {
        free_network(classify_network_);
    }

    void Classify::classify_image(image& in_darknet_image)
    {
        return forward(in_darknet_image);
    }

    image Classify::convert_image(const sensor_msgs::ImageConstPtr& msg)
    {
        if (msg->encoding != sensor_msgs::image_encodings::BGR8)
        {
            ROS_ERROR("Unsupported encoding");
            exit(-1);
        }

        auto data = msg->data;
        uint32_t height = msg->height, width = msg->width, offset = msg->step - 3 * width;
        uint32_t i = 0, j = 0;
        image im = make_image(width, height, 3);

        for (uint32_t line = height; line; line--)
        {
            for (uint32_t column = width; column; column--)
            {
                for (uint32_t channel = 0; channel < 3; channel++)
                    im.data[i + width * height * channel] = data[j++] / 255.;
                i++;
            }
            j += offset;
        }

        if (classify_network_->w == (int) width && classify_network_->h == (int) height)
        {
            return im;
        }
        image resized = resize_image(im, classify_network_->w, classify_network_->h);
        free_image(im);
        return resized;
    }

    void Classify::forward(image& in_darknet_image)
    {
        float * in_data = in_darknet_image.data;
        float *prediction = network_predict(classify_network_, in_data);

        if(classify_network_->hierarchy) hierarchy_predictions(prediction, classify_network_->outputs, classify_network_->hierarchy, 1, 1);
        // layer output_layer = classify_network_->layers[classify_network_->n - 1];
        //
        // output_layer.output = prediction;
        // int nboxes = 0;
        // int num_classes = output_layer.classes;
        // detection *darknet_detections = get_network_boxes(classify_network_, classify_network_->w, classify_network_->h, min_confidence_, .5, NULL, 0, &nboxes);
        //
        // do_nms_sort(darknet_detections, nboxes, num_classes, nms_threshold_);
        //
        // std::vector< RectClassScore<float> > detections;
        //
        // for (int i = 0; i < nboxes; i++)
        // {
        //     int class_id = -1;
        //     float score = 0.f;
        //     //find the class
        //     for(int j = 0; j < num_classes; ++j){
        //         if (darknet_detections[i].prob[j] >= min_confidence_){
        //             if (class_id < 0) {
        //                 class_id = j;
        //                 score = darknet_detections[i].prob[j];
        //             }
        //         }
        //     }
        //     //if class found
        //     if (class_id >= 0)
        //     {
        //         RectClassScore<float> detection;
        //
        //         detection.x = darknet_detections[i].bbox.x - darknet_detections[i].bbox.w/2;
        //         detection.y = darknet_detections[i].bbox.y - darknet_detections[i].bbox.h/2;
        //         detection.w = darknet_detections[i].bbox.w;
        //         detection.h = darknet_detections[i].bbox.h;
        //         detection.score = score;
        //         detection.class_type = class_id;
        //         //std::cout << detection.toString() << std::endl;
        //
        //         detections.push_back(detection);
        //     }
        // }
        // //std::cout << std::endl;
        // return detections;
    }
}  // namespace darknet

void rgbgr_image_y(image& im)
{
    int i;
    for(i = 0; i < im.w*im.h; ++i)
    {
        float swap = im.data[i];
        im.data[i] = im.data[i+im.w*im.h*2];
        im.data[i+im.w*im.h*2] = swap;
    }
}

image convert_ipl_to_image(const sensor_msgs::ImageConstPtr& msg)
{

    double image_ratio_;
    uint32_t image_top_bottom_border_;//black strips added to the input image to maintain aspect ratio while resizing it to fit the network input size
    uint32_t image_left_right_border_;

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(msg, "bgr8");//toCvCopy(image_source, sensor_msgs::image_encodings::BGR8);
    cv::Mat mat_image = cv_image->image;

    uint32_t network_input_width = yolo_detector_.get_network_width();
    uint32_t network_input_height = yolo_detector_.get_network_height();

    uint32_t image_height = msg->height,
            image_width = msg->width;

    IplImage ipl_image;
    cv::Mat final_mat;

    if (network_input_width!=image_width
        || network_input_height != image_height)
    {
        //final_mat = cv::Mat(network_input_width, network_input_height, CV_8UC3, cv::Scalar(0,0,0));
        image_ratio_ = (double ) network_input_width /  (double)mat_image.cols;

        cv::resize(mat_image, final_mat, cv::Size(), image_ratio_, image_ratio_);
        image_top_bottom_border_ = abs(final_mat.rows-network_input_height)/2;
        image_left_right_border_ = abs(final_mat.cols-network_input_width)/2;
        cv::copyMakeBorder(final_mat, final_mat,
                           image_top_bottom_border_, image_top_bottom_border_,
                           image_left_right_border_, image_left_right_border_,
                           cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    }
    else
        final_mat = mat_image;

    ipl_image = final_mat;

    unsigned char *data = (unsigned char *)ipl_image.imageData;
    int h = ipl_image.height;
    int w = ipl_image.width;
    int c = ipl_image.nChannels;
    int step = ipl_image.widthStep;
    int i, j, k;

    image darknet_image = make_image(w, h, c);

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                darknet_image.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    rgbgr_image_y(darknet_image);
    return darknet_image;
}

// int maximum_in_array(float* array, int N){
//   float big = array[0];
//   int idx = 0;
//   for(int i = 1; i < N; i++){
//     if(big < array[i]){
//       big = array[i];
//       idx = i;
//     }
//   }
//   return idx;
// }

image convert_ipl_to_image_classify(const sensor_msgs::ImageConstPtr& msg)
{

    double image_ratio_;
    uint32_t image_top_bottom_border_;//black strips added to the input image to maintain aspect ratio while resizing it to fit the network input size
    uint32_t image_left_right_border_;

    cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(msg, "bgr8");//toCvCopy(image_source, sensor_msgs::image_encodings::BGR8);
    cv::Mat mat_image = cv_image->image;

    uint32_t network_input_width = classifier_.get_network_width();
    uint32_t network_input_height = classifier_.get_network_height();

    uint32_t image_height = msg->height,
            image_width = msg->width;

    IplImage ipl_image;
    cv::Mat final_mat;

    if (network_input_width!=image_width
        || network_input_height != image_height)
    {
        //final_mat = cv::Mat(network_input_width, network_input_height, CV_8UC3, cv::Scalar(0,0,0));
        image_ratio_ = (double ) network_input_width /  (double)mat_image.cols;

        cv::resize(mat_image, final_mat, cv::Size(), image_ratio_, image_ratio_);
        image_top_bottom_border_ = abs(final_mat.rows-network_input_height)/2;
        image_left_right_border_ = abs(final_mat.cols-network_input_width)/2;
        cv::copyMakeBorder(final_mat, final_mat,
                           image_top_bottom_border_, image_top_bottom_border_,
                           image_left_right_border_, image_left_right_border_,
                           cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

    }
    else
        final_mat = mat_image;

    ipl_image = final_mat;

    unsigned char *data = (unsigned char *)ipl_image.imageData;
    int h = ipl_image.height;
    int w = ipl_image.width;
    int c = ipl_image.nChannels;
    int step = ipl_image.widthStep;
    int i, j, k;

    image darknet_image = make_image(w, h, c);

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                darknet_image.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    rgbgr_image_y(darknet_image);
    return darknet_image;
}

void image_callback(const sensor_msgs::ImageConstPtr& in_image_message)
{
    std::vector< RectClassScore<float> > detections;
    image darknet_image_ = {};
    darknet_image_ = convert_ipl_to_image(in_image_message);

    detections = yolo_detector_.detect(darknet_image_);

    //Check the if the network is able to score_threshold
    // float score_array[detections.size()];
    // for (unsigned int i = 0; i < detections.size(); ++i)
    //   score_array[i] = detections[i].score;
    //
    // int idx_max = maximum_in_array(score_array,detections.size());
    // std::cout<<"Score: "<<detections[idx_max].score<<" "<<"Class: "<<detections[idx_max].class_type<<std::endl;

    //Output messages for detect /darknet_ros

    dark_msgs::DetectArray output_detect_array;
    output_detect_array.header = in_image_message->header;
    for (unsigned int i = 0; i < detections.size(); ++i)
    {
        if(detections.size()>0)
        {
            dark_msgs::Detect output_detect;

            output_detect.x = detections[i].x;
            output_detect.y = detections[i].y;
            output_detect.w = detections[i].w;
            output_detect.h = detections[i].h;
            if (detections[i].x < 0)
                output_detect.x = 0;
            if (detections[i].y < 0)
                output_detect.y = 0;
            if (detections[i].w < 0)
                output_detect.w = 0;
            if (detections[i].h < 0)
                output_detect.h = 0;

            output_detect.score = detections[i].score;
            output_detect.class_id = detections[i].class_type;
            //std::cout << "x "<< rect.x<< " y " << rect.y << " w "<< rect.width << " h "<< rect.height<< " s " << rect.score << " c " << in_objects[i].class_type << std::endl;

            output_detect_array.objects.push_back(output_detect);

        }
    }

    detect_publish.publish(output_detect_array);
    free(darknet_image_.data);
}

void image_callback_classify(const sensor_msgs::ImageConstPtr& in_image_message)
{
    image darknet_image_ = {};
    darknet_image_ = convert_ipl_to_image_classify(in_image_message);

    classifier_.classify_image(darknet_image_);

    //Output messages for detect /darknet_ros

    // dark_msgs::DetectArray output_detect_array;
    // output_detect_array.header = in_image_message->header;
    // for (unsigned int i = 0; i < detections.size(); ++i)
    // {
    //     if(detections.size()>0)
    //     {
    //         dark_msgs::Detect output_detect;
    //
    //         output_detect.x = detections[i].x;
    //         output_detect.y = detections[i].y;
    //         output_detect.w = detections[i].w;
    //         output_detect.h = detections[i].h;
    //         if (detections[i].x < 0)
    //             output_detect.x = 0;
    //         if (detections[i].y < 0)
    //             output_detect.y = 0;
    //         if (detections[i].w < 0)
    //             output_detect.w = 0;
    //         if (detections[i].h < 0)
    //             output_detect.h = 0;
    //
    //         output_detect.score = detections[i].score;
    //         output_detect.class_id = detections[i].class_type;
    //         //std::cout << "x "<< rect.x<< " y " << rect.y << " w "<< rect.width << " h "<< rect.height<< " s " << rect.score << " c " << in_objects[i].class_type << std::endl;
    //
    //         output_detect_array.objects.push_back(output_detect);
    //
    //     }
    // }
    //
    // detect_publish.publish(output_detect_array);
    free(darknet_image_.data);
}

// void YoloNode::config_cb(const autoware_msgs::ConfigSsd::ConstPtr& param)
// {
//     score_threshold_ 	= param->score_threshold;
// }

void mySigintHandler(int sig)
{
  // Do some custom action.
  // For example, publish a stop message to some other nodes.

  // All the default sigint handler does is call shutdown()
  ros::shutdown();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "dark_ros");
  ROS_INFO("dl interface for ros is now running");
  ros::NodeHandle nh("~");
  signal(SIGINT, mySigintHandler);

  int flag = 1; //temprorily - detection 0, classification 1, segmentation 2
  //Network stuff
  //Here we have to give a condition, if the network chosen in YOLO
  //parameters which are to be converted with yaml

  if(flag==0){
    std::string image_raw_topic_str = "/usb_cam/image_raw", network_definition_file = "/home/ajwahir/ros_dl/src/dl_interface/darknet/cfg/yolo.cfg" ,pretrained_model_file = "/home/ajwahir/ros_dl/src/dl_interface/darknet/data/yolo.weights" ;
    float score_threshold_ = 0.5, nms_threshold_ = 0.45;
    ros::Subscriber subscriber_image_raw_;

    ROS_INFO("Initializing the network on Darknet...");
    yolo_detector_.load(network_definition_file, pretrained_model_file, score_threshold_, nms_threshold_);
    ROS_INFO("Initialization complete.");

    ROS_INFO("Subscribing to... %s", image_raw_topic_str.c_str());
    detect_publish = nh.advertise<dark_msgs::DetectArray>("detected_objects", 1);
    subscriber_image_raw_ = nh.subscribe(image_raw_topic_str, 1, image_callback);
  }
  else if(flag==1){
    std::string image_raw_topic_str = "/usb_cam/image_raw", network_definition_file = "/home/ajwahir/ros_dl/src/dl_interface/darknet/cfg/tiny.cfg" ,pretrained_model_file = "/home/ajwahir/ros_dl/src/dl_interface/darknet/data/tiny.weights" ;
    ros::Subscriber subscriber_image_raw_;

    ROS_INFO("Initializing the network on Darknet...");
    classifier_.load(network_definition_file, pretrained_model_file);
    ROS_INFO("Initialization complete.");

    ROS_INFO("Subscribing to... %s", image_raw_topic_str.c_str());
    // detect_publish = nh.advertise<dark_msgs::DetectArray>("detected_objects", 1);
    subscriber_image_raw_ = nh.subscribe(image_raw_topic_str, 1, image_callback);
  }


  ros::spin();

  return 0;
}
