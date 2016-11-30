#include <tracking/SkeletonTracker.h>
#include <iostream>

using namespace tracking;

SkeletonTracker::SkeletonTracker(IKinectSensor* sensor) :
	mTrackVelocity(false),
	m_pKinectSensor(sensor),
	m_pCoordinateMapper(nullptr),
	mCurrTime(0),
	mPrevTime(0)
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

void SkeletonTracker::TrackVelocity(bool active) {
	mTrackVelocity = active;
}

HRESULT SkeletonTracker::ExtractJoints(IBody* ppBodies[NR_USERS], INT64 timestamp)
{
	HRESULT hr = E_FAIL;

	// copy to buffer
	if (mTrackVelocity && timestamp > 0) {
		mUserIDsBuffered = mUserIDs;
		std::memcpy(mUserJointsBuffered, mUserJoints, NR_USERS * JointType_Count * sizeof(Joint));
		mPrevTime = mCurrTime;
		mCurrTime = timestamp;
	}

	// reset visible users
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

					// calculate joint velocity
					if (mTrackVelocity && mPrevTime > 0) {
						if (std::find(mUserIDsBuffered.begin(), mUserIDsBuffered.end(), iUser) != mUserIDsBuffered.end()) {
							for (int ijoint = 0; ijoint < JointType_Count; ijoint++) {
								INT64 dT = mCurrTime - mPrevTime;
								double vx = (mUserJoints[iUser][ijoint].Position.X - mUserJointsBuffered[iUser][ijoint].Position.X) / (double)dT;
								double vy = (mUserJoints[iUser][ijoint].Position.Y - mUserJointsBuffered[iUser][ijoint].Position.Y) / (double)dT;
								double vz = (mUserJoints[iUser][ijoint].Position.Z - mUserJointsBuffered[iUser][ijoint].Position.Z) / (double)dT;
								mJointVelocities[iUser][ijoint] = cv::Vec3d(vx,vy,vz);
							}
						}
					}

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
	joint_coords.clear();
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

int SkeletonTracker::GetFaceBoundingBoxes(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space, float box_size) const
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

	bounding_boxes.clear();
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

// TODO: fix boundaries
int SkeletonTracker::GetFaceBoundingBoxesRobust(std::vector<cv::Rect2f>& bounding_boxes, base::ImageSpace space, float box_size) const
{
	GetFaceBoundingBoxes(bounding_boxes, space, box_size);

	float xmin, xmax, ymin, ymax, width, height;

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

	// check for boundary overlapping values
	for(size_t i=0; i<bounding_boxes.size();i++)
	{
		xmin = (bounding_boxes[i].x > 0 ? bounding_boxes[i].x : 0);
		ymin = (bounding_boxes[i].y > 0 ? bounding_boxes[i].y : 0);
		width = (bounding_boxes[i].x + bounding_boxes[i].width > (srcWidth -1) ? srcWidth - bounding_boxes[i].x -1: bounding_boxes[i].width);
		height = (bounding_boxes[i].y + bounding_boxes[i].height > (srcHeight - 1) ? srcHeight - bounding_boxes[i].y -1: bounding_boxes[i].height);
		bounding_boxes[i].x = xmin;
		bounding_boxes[i].y = ymin;
		bounding_boxes[i].width = width;
		bounding_boxes[i].height = height;
	}
	return bounding_boxes.size();
}

int SkeletonTracker::GetUserSceneIDs(std::vector<int> &ids) const
{
	ids = mUserIDs;
	return mUserIDs.size();
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

	// draw velocities
	if (mTrackVelocity && mPrevTime > 0) {
		int nr_users = 0;
		int rect_height = 5;
		for (size_t i = 0; i < mUserIDs.size();i++) {
			nr_users++;
			// head velocity in m/s
			double vel = cv::norm(mJointVelocities[mUserIDs[i]][base::JointType_Head])* 10000000.;
			// draw velocit
			double min = 0.;
			double max = 3.;
			double px = 0.;
			px = (double)target.cols / max *vel;
			if (vel>max) {
				px = target.cols;
			}
			if (vel<min) {
				px = 0.;
			}
			//draw
			cv::rectangle(target, cv::Point(0, (nr_users - 1)* rect_height), cv::Point(px, nr_users * rect_height), cv::Scalar(0,0,255),3);
		}
	}


	// get face bounding boxes
	std::vector<cv::Rect2f> bounding_boxes;
	GetFaceBoundingBoxes(bounding_boxes, space);
	// draw bounding boxes
	for (size_t i = 0; i < bounding_boxes.size(); i++)
	{
		cv::rectangle(target, bounding_boxes[i], cv::Scalar(0, 0, 255), 2, cv::LINE_4);
	}

	return S_OK;
}

void SkeletonTracker::ExtractFacesPatches(cv::Mat img, int patch_size, std::vector<cv::Mat> &patches, std::vector<int> &user_ids) const
{
	// get face bounding boxes
	std::vector<cv::Rect2f> bounding_boxes;
	GetFaceBoundingBoxesRobust(bounding_boxes, base::ImageSpace_Color);

	for(int j = 0; j<mUserIDs.size();j++)
	{
		cv::Mat face = img(bounding_boxes[j]);
		if(face.cols == face.rows)
		{
			// resize
			cv::resize(face, face, cv::Size(patch_size, patch_size));
			// save id
			user_ids.push_back(mUserIDs[j]);
			// save face patch
			patches.push_back(face);
		}
	}
}
