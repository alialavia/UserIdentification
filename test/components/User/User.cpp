#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>
#include <user/User.h>

int main(int argc, char** argv)
{

	user::User* u = new user::User();
	user::IdentificationStatus is = user::IDStatus_WOOOOO;
	user::ActionStatus as;
	u->PrintMe(is, as);
	u->GetStatus2(is, as);
	u->PrintMe(is, as);

	return 0;
} 
