#include <iostream>
#include <string>

// include class header
#include <io/talker.h>

using namespace io;

Talker::Talker() {

}

void Talker::talk(){
	std::cout << "Hi there! I'm talking right now." << std::endl;
}

void Talker::say(std::string sentence){
	std::cout << sentence << std::endl;
}
