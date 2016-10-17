#ifndef IO__kinect
#define IO__kinect

#include <string>


#include<iostream>
#include<Windows.h>
// Kinect SDK 2
#include<kinect.h>

namespace io
{

class Kinect {
public:
	Kinect();
	void run_body_detection();
};

};

#endif