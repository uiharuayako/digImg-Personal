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
	// ��Ϊ���幦�ܵĽ��������ImgProcess���ṩͼ��������й���
	// ͼ�����ԭ�ļ�ӦΪraw��ʽ��ͼƬ��ʹ��opencv��ȡ����ʹ��opencv��غ���
private:
	string rawImgPath; // ԭͼ��·��
	Mat myImg; // opencv ��������ͼ��
	int imCols; // ����
	int imRows; // ����
	int type; // ��ʾͼ�����͵ı�����type��1=BIPͼ��2=BILͼ��3=BSQͼ��
	int pixels; // ��ʾͼ����n�������ϵ�����������nĬ��Ϊ3
	int channelNum; // ��������Ĭ��3
	void imgShowSave(Mat img, string title); // չʾͼƬ�����棬˽�к���
	unsigned char* pSrc; // ָ��ԭͼdata��ָ�룬�ڴ���ʱ����
	vector<double> hi1Template{ 0,-1,0,-1,5,-1,0,-1,0 };// ������˹����
	vector<double> hi2Template{ 1,-2,1,-2,5,-2,1,-2,1 };// ��һ�ָ�ͨ�˲�
	vector<double> hi3Template{ 0,0,-1,0,0,
								0,-1,-2,-1,0,
								-1,-2,16,-2,-1,
								0,-1,-2,-1,0,
								0,0,-1,0,0};// 5x5�ĸ�˹������˹
	vector<double> lowTemplate{ 0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111,0.11111111111 };
public:
	ImgProcess(string imgPath, int cols, int rows, int type); // ��ʼ��һ��ͼ���࣬��ɶ�ȡͼ���ܣ���ȡrawͼ��
	ImgProcess(string imgPath); // ��ʼ��һ��ͼ���࣬��ɶ�ȡͼ���ܣ���ȡ���������ͼ��
	void showRawImg(); // չʾδ�������ԭʼͼƬ
	// ����Ϊ����Ĺؼ�����
	// �����⣬ȫ�����
	// ����1��ͼ������㣺�Ҷ����Ա任
	// ���û��������Ա任��Ҫ�Ĳ������лҶ����Ա任
	void grayLinear(double k, double b); // ����1����ͼ����лҶ����Ա任����ѭ��ʽ y=k*x+b
	// ����2��ͼ��ֲ�������ͨ�˲�����ͨ�˲�����ֵ�˲�
	// д��һ���൱ͨ�õģ��ܴ����������ӵĺ���
	// pass�����е�myTemplate�������κ����ӣ�ֻҪд��һ�У�������vector�������ĳ��������ƽ��
	// �������������ָ�ͨģ�壬����3*3��һ��5*5��ʹ��ʱ���������м���ģ�峤�ȣ����Ƿǳ������
	Mat pass(vector<double> myTemplate); // ������������ڽ��и�ͨ�͵�ͨ�˲�
	void highPass(int type); // ����2.1����ͨ�˲�
	void lowPass(); // ����2.2����ͨ�˲�
	void medPass(); // ����2.3����ֵ�˲���sort��������
	// ѡ���⣬�����ȫ�����
	// ѡ��1��ͼ��ļ��δ���ƽ�ơ����š���ת
	// ƽ�Ƶ�ԭ����ͼ������λ�õı任
	// ���ţ���ת�����������˾���ı任
	void imgMove(int dx, int dy, int type); // ѡ��1.1��ͼ��ƽ�ƣ�dx��dy�����أ�typeΪ0��ԭͼ���С���䣬Ϊ1��ԭͼ���С�ı�
	void imgZoom(double xRate, double yRate); // ѡ��1.2��ͼ������
	void imgRotate(double angle); // ѡ��1.3 ͼ����ת��˳ʱ����ת��angleΪ�Ƕ��ƵĽǶ�
	// ѡ��2��ͼ���ֵ����״̬�����жϷ�����
	// ͼ���ֵ���Ĺؼ����ҵ���ֵ����ʹ���������㷨���ж���ֵ
	// ��һ�֣��Ǳ�֤����ֵ���ֵ�����Ҷȼ�֮�䷽������
	// �ڶ����Ƿǳ�������Ч���Ϻõ�OTSU�㷨
	void imgBinRatio(); // ѡ��2.1��ͼ���ֵ������������
	void imgBinOTSU(); // ѡ��2.2��ͼ���ֵ����otsu��
	// ѡ��3������ͼ�������غ���������
	// �����������ɵ������ͼ��������֣����ǳ�����㷨������ζ�μ�飬�Ҿ�����ȷʵû�����
	// �������ٶ���ĺ������㷨�����ӶȾ͸ߵ����ף�
	void imgAutoCorr(); // ѡ��3��ͼ������غ������������Ҷ�ͼ��
	// ѡ��4��α��ɫ��ǿ
	// ʹ�ù�ʽ��ԭ�Ҷ�ͼ����зֶα任
	void bin2Color(); // ѡ��4��α��ɫ��ǿ
	// ѡ��5��ɫ��ƽ��
	// ��������ɫ��ƽ�⣬��һ���ǰ�ƽ���㷨
	// �ڶ����ǰ����û������RGBֵ����ƽ��
	void whiteBalance(); // ѡ��5��ɫ��ƽ�⣺��ƽ��
	void colorBalance(int deltaR, int deltaG, int deltaB); // ѡ��5��ɫ��ƽ�⣺ɫ��ƽ��
};

