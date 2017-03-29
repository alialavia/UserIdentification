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

	c1.Insert({ 4 }, { "asadf" });
	c1.Insert({ 4 }, { "asadf" });
	c1.Insert({ 5 }, { "asadf" });
	c1.InsertIntersection({ 1, 4 }, { "hey", "hello" });
	c1.Print();

	// second set
	c2.Insert({ 7 }, { "hello7" });
	c2.Insert({ 10 }, { "hey" });

	// no merge
	c1.MergeIntersection(c2);
	c1.Print();
	// merge
	c2.Insert({ 1 }, { "grüezi" });
	c1.MergeIntersection(c2);
	c1.Print();

}

void TestListSet()
{
	math::UniqueSetAssoc<int, std::string> c1;
	math::UniqueSetAssoc<int, std::string> c2;
	math::UniqueSetAssoc<int, std::string> c3;

	c1.Insert({1,2}, {"a", "b"});
	c2.Insert({1,2,3}, {"a", "b", "c"});
	c3.Insert({2,3}, {"d", "e"});

	math::UniqueSetList<int, std::string> set;

	set.Update(1, 2, "a", "b");
	set.Print();
	set.Update(3, 4, "c", "d");
	set.Print();
	set.Update(3, 2, "aa32_1", "aa32_2");
	set.Print();
}


int main(int argc, char** argv)
{


	//TestUniqueList();

	TestListSet();

	return 0;
} 
