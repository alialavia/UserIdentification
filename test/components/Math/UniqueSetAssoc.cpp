#include <gflags/gflags.h>
#include <iostream>
#include <opencv2/core/cvdef.h>
#include <typeindex>
#include <typeinfo>
#include <map>
#include <opencv2/opencv.hpp>
#include <Math/Math.h>


void TestUniqueList()
{
	math::UniqueSetAssoc<int, std::string> c1;
	math::UniqueSetAssoc<int, std::string> c2;
	math::UniqueSetAssoc<int, std::string> c3;

	c1.Insert(4,"asadf");
	c1.Insert(4,"asadf");
	c1.Insert(5,"asadf");
	c1.InsertIntersection({ 1, 4 }, { "hey", "hello" });
	c1.Print();


	// second set
	c2.Insert(7, "hello7");
	c2.Insert(10, "hey");

	// no merge
	c1.MergeIntersection(c2);
	c1.Print();
	// merge
	c2.Insert(1, "grüezi");
	c1.MergeIntersection(c2);
	c1.Print();

}


int main(int argc, char** argv)
{


	TestUniqueList();

	return 0;
} 
