#ifndef IO__talker
#define IO__talker

#include <string>

namespace io
{
	class Talker {
		public:
		Talker();
		void Talker::talk();
		void Talker::say(std::string sentence);
	};
};

#endif