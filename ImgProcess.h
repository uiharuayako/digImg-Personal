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
	vector<double> lowTemplate{ 0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111 };
public:
	ImgProcess(string imgPath, int cols, int rows, int type); // 初始化一个图像类，完成读取图像功能，读取raw图像
	ImgProcess(string imgPath); // 初始化一个图像类，完成读取图像功能，读取其他种类的图像
	void showRawImg(); // 展示未经处理的原始图片
	// 以下为任务的关键函数
	// 必做题，全部完成
	// 必做1：图像点运算：灰度线性变换
	// 由用户输入线性变换需要的参数进行灰度线性变换
	void grayLinear(double k, double b); // 必做1：对图像进行灰度线性变换，遵循公式 y=k*x+b
	// 必做2：图像局部处理：高通滤波、低通滤波、中值滤波
	// 写了一个相当通用的，能处理任意算子的函数
	// pass函数中的myTemplate可以是任何算子，只要写成一行，储存在vector里，长度是某个奇数的平方
	// 程序内置了三种高通模板，两个3*3，一个5*5，使用时函数会自行计算模板长度，还是非常方便的
	Mat pass(vector<double> myTemplate); // 这个函数被用于进行高通和低通滤波
	void highPass(int type); // 必做2.1：高通滤波
	void lowPass(); // 必做2.2：低通滤波
	void medPass(); // 必做2.3：中值滤波，sort函数排序
	// 选做题，五道题全部完成
	// 选做1：图像的几何处理：平移、缩放、旋转
	// 平移的原理是图像像素位置的变换
	// 缩放，旋转，则是利用了矩阵的变换
	void imgMove(int dx, int dy, int type); // 选做1.1：图像平移（dx，dy）像素，type为0，原图像大小不变，为1，原图像大小改变
	void imgZoom(double xRate, double yRate); // 选做1.2：图像缩放
	void imgRotate(double angle); // 选做1.3 图像旋转，顺时针旋转，angle为角度制的角度
	// 选做2：图像二值化：状态法及判断分析法
	// 图像二值化的关键是找到阈值，我使用了两种算法来判断阈值
	// 第一种，是保证被阈值区分的两组灰度级之间方差比最大
	// 第二种是非常常用且效果较好的OTSU算法
	void imgBinRatio(); // 选做2.1：图像二值化，方差比最大法
	void imgBinOTSU(); // 选做2.2：图像二值化，otsu法
	// 选做3：纹理图像的自相关函数分析法
	// 做出来，生成的自相关图像结果很奇怪，但是程序的算法经过多次多次检查，我觉得是确实没问题的
	// （计算速度真的很慢，算法本身复杂度就高的离谱）
	void imgAutoCorr(); // 选做3：图像自相关函数：仅分析灰度图像
	// 选做4：伪彩色增强
	// 使用公式对原灰度图像进行分段变换
	void bin2Color(); // 选做4：伪彩色增强
	// 选做5：色彩平衡
	// 做了两种色彩平衡，第一种是白平衡算法
	// 第二种是按照用户定义的RGB值进行平衡
	void whiteBalance(); // 选做5：色彩平衡：白平衡
	void colorBalance(int deltaR, int deltaG, int deltaB); // 选做5：色彩平衡：色彩平衡
};

