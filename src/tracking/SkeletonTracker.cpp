#include <tracking/SkeletonTracker.h>
#include <iostream>

using namespace tracking;

SkeletonTracker::SkeletonTracker(IKinectSensor* sensor) :
	m_pKinectSensor(sensor),
	m_pCoordinateMapper(nullptr)
{
}

SkeletonTracker::~SkeletonTracker()
{
	SafeRelease(m_pCoordinateMapper);
}


HRESULT SkeletonTracker::Init()
{
	HRESULT hr = E_FAIL;
	if (m_pKinectSensor)
	{
		hr = m_pKinectSensor->get_CoordinateMapper(&m_pCoordinateMapper);
	}
	return hr;
}

HRESULT SkeletonTracker::ExtractJoints(IBody* ppBodies[NR_USERS])
{
	HRESULT hr = E_FAIL;
	mUserIDs.clear();

	for (int iUser = 0; iUser < NR_USERS; ++iUser)
	{
		IBody* pBody = ppBodies[iUser];
		if (pBody)
		{
			BOOLEAN bTracked = false;
			hr = pBody->get_IsTracked(&bTracked);

			if (SUCCEEDED(hr) && bTracked)
			{
				Joint joints[JointType_Count]; // joints in body space

				// get joints
				hr = pBody->GetJoints(_countof(joints), joints);

				if (SUCCEEDED(hr))
				{
					/*
					UINT64 trackingId;
					hr = pBody->get_TrackingId(&trackingId);
					std::cout << "Userid: " << trackingId << "\n";
					*/

					// save user id
					mUserIDs.push_back(iUser);

					// save skeleton
					std::memcpy(mUserJoints[iUser], joints, JointType_Count * sizeof (Joint));
				}
			}
		}
	}

	return hr;
}


int SkeletonTracker::GetActiveBoundingBoxes(std::vector<cv::Rect2d>& boxes, std::vector<int>& user_ids) const
{
	HRESULT hr = E_FAIL;

	return hr;
}


void SkeletonTracker::reset()
{
	mBoundingBoxes.clear();
	mUserIDs.clear();
}

/*
 * Joint indices:
 * - JointType_Head
 * - ...
 */
// TODO: fix position scaling
int SkeletonTracker::GetJointsColorSpace(std::vector<std::vector<cv::Point2f>>& joints_colorspace, int srcWidth, int srcHeight, int outputWidth, int outputHeight) const
{
	for (size_t j = 0; j < mUserIDs.size(); j++)
	{

		size_t iUser = mUserIDs[j];
		ColorSpacePoint colorspace_pt = {0};
		CameraSpacePoint cameraspace_pt;
		std::vector <cv::Point2f> color_coordinates_cv;

		// map body space to color space
		for (int j = 0; j < JointType_Count; ++j)
		{
			cameraspace_pt = mUserJoints[iUser][j].Position;
			m_pCoordinateMapper->MapCameraPointToColorSpace(cameraspace_pt, &colorspace_pt);

			//std::cout << "GetJointsColorSpace P1(" << colorspace_pt.X << ", " << colorspace_pt.Y << ")\n";
			//std::cout << "GetJointsColorSpace P2(" << cameraspace_pt.X << ", " << cameraspace_pt.Y << ")\n";
			// scale to output size
			float screenPointX = static_cast<float>(colorspace_pt.X * outputWidth) / outputHeight;
			float screenPointY = static_cast<float>(colorspace_pt.Y * outputHeight) / srcHeight;

			color_coordinates_cv.push_back(cv::Point2f(screenPointX, screenPointY));
		}

		// store
		joints_colorspace.push_back(color_coordinates_cv);
	}



	return mUserIDs.size();
}


// -------------------- User Tracker

UserTracker::UserTracker(IKinectSensor* sensor):
	pKinectSensor(sensor),
	pSkeletonTracker(nullptr)
{
}

UserTracker::~UserTracker()
{
}

HRESULT UserTracker::Init()
{
	// initialize tracking routines
	pSkeletonTracker = new SkeletonTracker(pKinectSensor);
	HRESULT hr = pSkeletonTracker->Init();

	return hr;
}
