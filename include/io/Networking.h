#ifndef IO_NETWORKING_H_
#define IO_NETWORKING_H_
#include <cstdint>
#include <windows.h>

namespace cv {
	class Mat;
}

namespace io
{
	class TCPClient
	{
	public:
		TCPClient();
		~TCPClient();
		bool Connect(char* host_name = "127.0.0.1", int host_port = 80) const;
		void Close();
		bool SendKeyboard();
		// absolute limitation on TCP packet size is 64K (65535 bytes)
		void SendRGBTestImage(int size = 100);
		int SendImage(cv::Mat img);

		void WaitForResponse();
		// range: -127 .. 127
		int SendRequestID(char id);
		int SendMessageSize(uint32_t size);
		int ReceiveMessage(int s, char *buf, int *len);


		// primitives handling
		unsigned int ReceiveUnsignedInt();
		unsigned short int ReceiveUnsignedShortInt();


	private:
		bool OpenSocket();
		int mSocketID;
	};
}

#endif