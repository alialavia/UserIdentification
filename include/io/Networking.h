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

	private:
		bool OpenSocket();
		int mSocketID;
	};
}