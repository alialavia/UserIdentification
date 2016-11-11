#ifndef IO_NETWORKING_H_
#define IO_NETWORKING_H_
#include <cstdint>
#include <windows.h>
#include <vector>

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
		bool Connect(char* host_name = "127.0.0.1", int host_port = 80);
		void Reconnect();
		void Close();

		/// <summary>
		/// Sends a RGB image. The RGB color system uses 3 channels (Red, Green, Blue) 8 bits = 1 byte each, which have (unsigned) integer values from 0 to 255.
		/// </summary>
		/// <param name="img">The img.</param>
		/// <returns>Number of bytes sent</returns>
		int SendRGBImage(const cv::Mat &img);
		void SendRGBTestImage(int size = 100);

		// range: -127 .. 127
		int SendChar(char id);
		int SendUInt(uint32_t size);
		bool SendKeyboard();

		// receive
		int ReceiveMessage(int socket_id, char *buf, int *len);
		int ReceiveRGBImage(cv::Mat &output, int img_width);

		// primitives handling
		unsigned int ReceiveUnsignedInt();
		unsigned short int ReceiveUnsignedShortInt();

		// ------ deprecated
		int SendImageWithLength(const cv::Mat &img);
		int SendImageBatchWithLength(const std::vector<cv::Mat> &images);

	private:
		bool OpenSocket();
		int mSocketID;
		char* mHostName;
		int mHostPort;
	};
}

#endif