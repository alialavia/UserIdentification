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

void SkeletonTracker::reset()
{
	mBoundingBoxes.clear();
	mUserIDs.clear();
}

// --------------- data access

int SkeletonTracker::GetJoints(std::vector<std::vector<cv::Point2f>>& joint_coords, DWORD joints, base::ImageSpace space,
	int outputWidth, int outputHeight) const
{
	ColorSpacePoint colorspace_pt = { 0 };
	DepthSpacePoint depthspace_pt = { 0 };
	CameraSpacePoint cameraspace_pt = { 0 };
	bool map_to_colorspace = (base::ImageSpace_Color & space) == base::ImageSpace_Color;
	float screenPointX, screenPointY;
	int srcWidth, srcHeight;

	// original stream size
	if (map_to_colorspace)
	{
		srcWidth = base::StreamSize_WidthColor;
		srcHeight = base::StreamSize_HeightColor;
	}
	else
	{
		srcWidth = base::StreamSize_WidthDepth;
		srcHeight = base::StreamSize_HeightDepth;
	}

	for (size_t i = 0; i < mUserIDs.size(); i++)
	{
		size_t iUser = mUserIDs[i];

		std::vector <cv::Point2f> color_coordinates_cv;

		int byte = 1;


		for (int j = 0; j < base::JointType_Count; ++j)
		{
			// joint selection
			if ((byte & joints) == byte) {
				cameraspace_pt = mUserJoints[iUser][j].Position;

				if (map_to_colorspace)
				{
					m_pCoordinateMapper->MapCameraPointToColorSpace(cameraspace_pt, &colorspace_pt);
					screenPointX = colorspace_pt.X;
					screenPointY = colorspace_pt.Y;
				}
				else
				{
					m_pCoordinateMapper->MapCameraPointToDepthSpace(cameraspace_pt, &depthspace_pt);
					screenPointX = depthspace_pt.X;
					screenPointY = depthspace_pt.Y;
				}

				// scaling and conversion
				screenPointX = static_cast<float>(screenPointX * outputWidth) / srcWidth;
				screenPointY = static_cast<float>(screenPointY * outputHeight) / srcHeight;
				color_coordinates_cv.push_back(cv::Point2f(screenPointX, screenPointY));
			}
			byte *= 2;
		}

		// store
		joint_coords.push_back(color_coordinates_cv);
	}

	return mUserIDs.size();
}

int SkeletonTracker::GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space,
	int outputWidth, int outputHeight, float box_size) const
{
	int srcWidth, srcHeight;
	if ((base::ImageSpace_Color & space) == base::ImageSpace_Color)
	{
		srcWidth = base::StreamSize_WidthColor;
		srcHeight = base::StreamSize_HeightColor;
	}
	else
	{
		srcWidth = base::StreamSize_WidthDepth;
		srcHeight = base::StreamSize_HeightDepth;
	}

	ColorSpacePoint p1c, p2c, p3c, p4c;
	CameraSpacePoint head_center, p1, p2, p3, p4;
	std::vector <cv::Point2f> color_coordinates_cv;
	cv::Rect2f bounding_box;

	for (size_t j = 0; j < mUserIDs.size(); j++)
	{
		size_t iUser = mUserIDs[j];

		// head
		head_center = mUserJoints[iUser][3].Position;

		// center between head and neck
		// CameraSpacePoint neck;
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

	return bounding_boxes.size();

}

// --------------- drawing methods

HRESULT SkeletonTracker::RenderFaceBoundingBoxes(cv::Mat &target, base::ImageSpace space) const
{
	int srcWidth, srcHeight;
	if ((base::ImageSpace_Color & space) == base::ImageSpace_Color)
	{
		srcWidth = base::StreamSize_WidthColor;
		srcHeight = base::StreamSize_HeightColor;
	}
	else
	{
		srcWidth = base::StreamSize_WidthDepth;
		srcHeight = base::StreamSize_HeightDepth;
	}

	int output_width, output_height;
	output_width = target.cols;
	output_height = target.rows;

	// get joints
	std::vector<std::vector<cv::Point2f>> user_joints;
	int nr_users = 0;
	static const DWORD joints =
		base::JointType_Head
		| base::JointType_Neck;

	nr_users = GetJoints(user_joints, joints, space, output_width, output_height);
	// draw joints
	for (size_t i = 0; i < user_joints.size(); i++)
	{
		for (size_t j = 0; j<user_joints[i].size(); j++)
		{
			cv::circle(target, user_joints[i][j], 4, cv::Scalar(0, 255, 0), cv::LINE_4);
		}
	}

	// get face bounding boxes
	std::vector<cv::Rect2f> bounding_boxes;
	GetFaceBoundingBoxes(bounding_boxes, space, output_width, output_height);
	// draw bounding boxes
	for (size_t i = 0; i < bounding_boxes.size(); i++)
	{
		cv::rectangle(target, bounding_boxes[i], cv::Scalar(0, 0, 255), 2, cv::LINE_4);
	}

	return S_OK;
}

// --------------- User Tracker

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
