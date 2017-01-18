#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <user/User.h>

int main(int argc, char** argv)
{

	user::User* u = new user::User();
	user::IdentificationStatus is = user::IDStatus_Identified;
	user::ActionStatus as;
	u->SetUserID(13, "");

	int u_id;
	std::string u_name;
	u->GetUserID(u_id, u_name);

	std::cout << u_id << std::endl;

	return 0;
} 
