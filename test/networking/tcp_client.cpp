#include <io/Networking.h>
#include <gflags/gflags.h>
#include <iostream>

DEFINE_int32(port, 80, "Server port");

int main(int argc, char** argv)
{

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	io::TCPClient c;
	c.Connect("127.0.0.1", FLAGS_port);

	c.SendRequestID(1);
	c.SendMessageSize(92349);
	std::cout <<  c.WaitForIDResponse();

	//c.SendRGBTestImage(100);
	//std::cout << "--- image send, now waiting to receive\n";
	//c.WaitForResponse();


} 