#ifndef USER_USER_H_
#define USER_USER_H_
#include <string>

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
		User() : mUserID(-1), mUserNiceName(""), mIDStatus(IDStatus_Unknown)
		{

		}

		~User();
		void SetUserID(int id, std::string nice_name);
		void SetIDStatus(enum IdentificationStatus status);
		void SetFaceBoundingBox(cv::Rect2f bb);
		enum IdentificationStatus GetIDStatus();
		void GetUserID(int& id, std::string& nice_name) const;
		cv::Rect2f GetFaceBoundingBox();

	private:
		int mUserID;
		std::string mUserNiceName;
		enum IdentificationStatus mIDStatus;
		cv::Rect2f mFaceBoundingBox;

	};


}

#endif