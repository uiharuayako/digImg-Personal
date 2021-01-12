// digImg-Personal.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("ik_beijing_c.bmp");

	// unsigned char* ptr = img.data;

	// 获取图像的宽高
	int height, width;
	height = img.rows;
	width = img.cols;

	// 创建灰度图
	Mat grayImg = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b px = img.at<Vec3b>(i, j);
			grayImg.at<uchar>(i, j) = px[0] / 3 + px[1] / 3 + px[2] / 3;
		}
	}

	// 输出
	namedWindow("灰度图", 0);
	imshow("灰度图", grayImg);
	waitKey();

	getchar();
	return 0;
}
