#ifndef USER_USER_H_
#define USER_USER_H_

namespace user
{

	enum IdentificationStatus
	{
		IDStatus_Unknown = 0,
		IDStatus_Identified = 1,
		IDStatus_Pending = 2
	};


	class User {

	public:
		User() : mUserID(-1), mIDStatus(IDStatus_Unknown)
		{

		}

		~User();
		void SetUserID(int id);
		void SetIDStatus(enum IdentificationStatus status);
		void SetFaceBoundingBox(cv::Rect2f bb);
		enum IdentificationStatus GetIDStatus();
		int GetUserID();
		cv::Rect2f GetFaceBoundingBox();

	private:
		int mUserID;
		enum IdentificationStatus mIDStatus;
		cv::Rect2f mFaceBoundingBox;

	};


}

#endif