#include <vector>
#include <gflags/gflags.h>
#include <iostream>
#include <algorithm>
#include <math\Math.h>


void integer_test() {
	math::SequentialContainer<int> c;

	for (int i = 0; i < 10; i++) {
		c.push(i);
	}

	int duplicate = 4;
	c.push(duplicate);

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
}

void reference_test() {

	int a, b, c;
	a = 1;
	b = 2;
	c = 3;

	math::SequentialContainer<int*> cont;
	cont.push(&a);
	cont.push(&b);
	cont.push(&c);

	cont.erase(&b);

	while (!cont.empty()) {
		std::cout << *cont.front() << std::endl;
		cont.pop();
	}
}


int main(int argc, char** argv)
{
	reference_test();
	return 0;
} 
