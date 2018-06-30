#include <ros/ros.h>
#include <iostream>
#include <signal.h>
#include <stdio.h>

void mySigintHandler(int sig)
{
  // Do some custom action.
  // For example, publish a stop message to some other nodes.

  // All the default sigint handler does is call shutdown()
  ros::shutdown();
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "twist_stamped_usb");
  ROS_INFO("Twist to serial driver is now running");
  ros::NodeHandle nh("~");
  signal(SIGINT, mySigintHandler);


  while (ros::ok())
  {


      ros::spinOnce();

  }

  return 0;
}
