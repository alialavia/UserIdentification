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


//joint selection:
//static const DWORD joints =
//base::JointType_Head
//| base::JointType_Neck;
int SkeletonTracker::GetJointsColorSpace(std::vector<std::vector<cv::Point2f>>& joints_colorspace, const DWORD joints, int srcWidth, int srcHeight, int outputWidth, int outputHeight) const
{
	for (size_t j = 0; j < mUserIDs.size(); j++)
	{
		size_t iUser = mUserIDs[j];
		ColorSpacePoint colorspace_pt = {0};
		CameraSpacePoint cameraspace_pt;
		std::vector <cv::Point2f> color_coordinates_cv;

		int byte = 1;
		// map body space to color space
		for (int j = 0; j < base::JointType_Count; ++j)
		{
			// if extracted
			if ((byte & joints) == byte) {
				cameraspace_pt = mUserJoints[iUser][j].Position;
				m_pCoordinateMapper->MapCameraPointToColorSpace(cameraspace_pt, &colorspace_pt);
				// scale to output size
				float screenPointX = static_cast<float>(colorspace_pt.X * outputWidth) / srcWidth;
				float screenPointY = static_cast<float>(colorspace_pt.Y * outputHeight) / srcHeight;
				color_coordinates_cv.push_back(cv::Point2f(screenPointX, screenPointY));
			}
			byte *= 2;
		}

		// store
		joints_colorspace.push_back(color_coordinates_cv);
	}

	return mUserIDs.size();
}


int SkeletonTracker::GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, int srcWidth, int srcHeight, int outputWidth, int outputHeight, float box_size) const
{
	
	ColorSpacePoint colorspace_pt, p1c, p2c, p3c, p4c;
	CameraSpacePoint head_center, neck, p1, p2, p3, p4;
	std::vector <cv::Point2f> color_coordinates_cv;
	cv::Rect2f bounding_box;
	
	for (size_t j = 0; j < mUserIDs.size(); j++)
	{
		size_t iUser = mUserIDs[j];

		// head
		head_center = mUserJoints[iUser][3].Position;

		// center between head and neck
		// neck = mUserJoints[iUser][2].Position;
		//head_center.X = (head_center.X + neck.X) / 2.;
		//head_center.Y = (head_center.Y + neck.Y) / 2.;

		p1 = head_center;
		p2 = head_center;
		p3 = head_center;
		p4 = head_center;
		p1.X += box_size / 2.;
		p1.Y += box_size / 2.;
		p2.X += box_size / 2.;
		p2.Y -= box_size / 2.;
		p3.X -= box_size / 2.;
		p3.Y -= box_size / 2.;
		p4.X -= box_size / 2.;
		p4.Y += box_size / 2.;

		//m_pCoordinateMapper->MapCameraPointToColorSpace(cameraspace_pt, &colorspace_pt);
		m_pCoordinateMapper->MapCameraPointToColorSpace(p1, &p1c);
		m_pCoordinateMapper->MapCameraPointToColorSpace(p2, &p2c);
		m_pCoordinateMapper->MapCameraPointToColorSpace(p3, &p3c);
		m_pCoordinateMapper->MapCameraPointToColorSpace(p4, &p4c);

		float head_size_colorspace = abs(p4c.X - p1c.X);
		bounding_box = cv::Rect2f(p4c.X, p4c.Y, head_size_colorspace, head_size_colorspace);

		// store
		bounding_boxes.push_back(bounding_box);
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
