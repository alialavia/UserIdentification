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
					std::memcpy(mUserJoints[iUser], joints, sizeof mUserJoints[iUser]);
				}
			}
		}
	}

	return hr;
}


int SkeletonTracker::GetActiveBoundingBoxes(std::vector<cv::Rect2d>& boxes, std::vector<int>& user_ids) const
{
	HRESULT hr = E_FAIL;

	//for (int i = 0; i < NR_USERS; ++i)
	//{
	//	IBody* pBody = ppBodies[i];
	//	if (pBody)
	//	{
	//		BOOLEAN bTracked = false;
	//		hr = pBody->get_IsTracked(&bTracked);

	//		if (SUCCEEDED(hr) && bTracked)
	//		{
	//			Joint joints[JointType_Count]; // joints in body space
	//			ColorSpacePoint color_coordinates[JointType_Count];

	//			// get joints
	//			hr = pBody->GetJoints(_countof(joints), joints);

	//			if (SUCCEEDED(hr))
	//			{
	//				for (int j = 0; j < _countof(joints); ++j)
	//				{
	//					//jointPoints[j] = BodyToScreen(joints[j].Position, width, height);
	//					m_pCoordinateMapper->MapCameraPointToColorSpace(joints[j].Position, &color_coordinates[j]);

	//				}

	//				// save

	//			}
	//		}
	//	}
	//}

	return hr;
}

//int SkeletonTracker::GetActiveBoundingBoxes(std::vector<cv::Rect2d>& boxes, std::vector<int>& user_ids) const
//{
//	boxes = mBoundingBoxes;
//	user_ids = mUserIDs;
//	return mUserIDs.size();
//}

void SkeletonTracker::reset()
{
	mBoundingBoxes.clear();
	mUserIDs.clear();
}


int SkeletonTracker::GetJointsColorSpace(std::vector<cv::Point2i*>& joints_colorspace, int* nr_users = nullptr)
{
	for (size_t iUser = 0; iUser < mUserIDs.size(); iUser++)
	{
		ColorSpacePoint color_coordinates[JointType_Count];
		cv::Point2i color_coordinates_cv[JointType_Count];

		// map body space to color space
		for (int j = 0; j < JointType_Count; ++j)
		{
			m_pCoordinateMapper->MapCameraPointToColorSpace(mUserJoints[iUser][j].Position, &color_coordinates[j]);
			color_coordinates_cv[j] = cv::Point2i(color_coordinates[j].X, color_coordinates[j].Y);
		}

		// store
		joints_colorspace.push_back(color_coordinates_cv);
	}

	if (nr_users != nullptr)
	{
		*nr_users = mUserIDs.size();
	}


	return mUserIDs.size();
}

int SkeletonTracker::GetJointsColorSpace(std::vector<ColorSpacePoint*>& joints_colorspace, int* nr_users = nullptr)
{
	for (size_t iUser = 0; iUser < mUserIDs.size(); iUser++)
	{
		ColorSpacePoint color_coordinates[JointType_Count];

		// map body space to color space
		for (int j = 0; j < JointType_Count; ++j)
		{
			m_pCoordinateMapper->MapCameraPointToColorSpace(mUserJoints[iUser][j].Position, &color_coordinates[j]);
		}

		// store
		joints_colorspace.push_back(color_coordinates);
	}

	if (nr_users != nullptr)
	{
		*nr_users = mUserIDs.size();
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
