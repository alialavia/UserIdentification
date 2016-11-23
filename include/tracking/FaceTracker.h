#ifndef TRACKING_FACETRACKER_H_
#define TRACKING_FACETRACKER_H_
#include <Kinect.Face.h>

namespace tracking
{
	
	class FaceTracker
	{
	public:
		FaceTracker(IKinectSensor* sensor):
		m_pKinectSensor(sensor)
		{
			
		}

	private:
		IKinectSensor* m_pKinectSensor;


	};

}

#endif