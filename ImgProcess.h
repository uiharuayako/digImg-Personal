#pragma once
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <numeric>
#include <algorithm>
using namespace std;
using namespace cv;
class ImgProcess
{
	// 作为整体功能的解决方案，ImgProcess类提供图像处理的所有功能
	// 图像处理的原文件应为raw格式的图片，使用opencv读取后不再使用opencv相关函数
private:
	string rawImgPath; // 原图像路径
	Mat myImg; // opencv 读出来的图像
	int imCols; // 列数
	int imRows; // 行数
	int type; // 表示图像类型的变量，type：1=BIP图像，2=BIL图像，3=BSQ图像
	int pixels; // 表示图像在n个波段上的像素总量，n默认为3
	int channelNum; // 波段数，默认3
	void imgShowSave(Mat img, string title); // 展示图片并保存，私有函数
	unsigned char* pSrc; // 指向原图data的指针，在处理时常用
	vector<double> hi1Template{ 0,-1,0,-1,5,-1,0,-1,0 };// 拉普拉斯算子
	vector<double> hi2Template{ 1,-2,1,-2,5,-2,1,-2,1 };// 另一种高通滤波
	vector<double> hi3Template{ 0,0,-1,0,0,
								0,-1,-2,-1,0,
								-1,-2,16,-2,-1,
								0,-1,-2,-1,0,
								0,0,-1,0,0};// 5x5的高斯拉普拉斯
	vector<double> lowTemplate{ 1 / 9,1 / 9,1 / 9,1 / 9,1 / 9,1 / 9,1 / 9,1 / 9,1 / 9 };
public:
	ImgProcess(string imgPath, int cols, int rows, int type); // 初始化一个图像类，完成读取图像功能，读取raw图像
	ImgProcess(string imgPath); // 初始化一个图像类，完成读取图像功能，读取其他种类的图像
	void showRawImg(); // 展示未经处理的原始图片
	void grayLinear(double k, double b); // 必做1：对图像进行灰度线性变换，遵循公式 y=k*x+b
	Mat pass(vector<double> myTemplate); // 这个函数被用于进行高通和低通滤波，接受一个3*3滤波模板作为参数。滤波模板是个长度为9的double数组
	void highPass(int type); // 必做2.1：高通滤波，提供两种模板，对应书本上的H1和H2矩阵
	void lowPass(); // 必做2.2：低通滤波
	void medPass(); // 必做2.3：中值滤波
	void imgMove(int dx, int dy, int type); // 选做1.1：图像平移（dx，dy）像素，type为0，原图像大小不变，为1，原图像大小改变
	void imgZoom(double xRate, double yRate); // 选做1.2：图像缩放
	void imgRotate(double angle); // 选做1.3 图像旋转，顺时针旋转
	void imgBinRatio(); // 选做2.1：图像二值化，方差比最大法
	void imgBinOTSU(); // 选做2.2：图像二值化，otsu法
	void bin2Color(); // 选做4：伪彩色增强
	void colorBalance(); // 选做5：色彩平衡：白平衡
};

