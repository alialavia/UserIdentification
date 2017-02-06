#include <vector>
#include <gflags/gflags.h>
#include <iostream>
#include <algorithm>
#include <math\Math.h>

int main(int argc, char** argv)
{
	math::SequentialContainer<int> c;

	for (int i = 0; i < 10;i++) {
		c.push(i);
	}

	// element not in container
	int to_erase = 22;
	c.erase(to_erase);

	// delete existing element
	to_erase = 4;
	c.erase(to_erase);

	while (!c.empty()) {
		std::cout << c.front() << std::endl;
		c.pop();
	}

	return 0;
} 
