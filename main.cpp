#include "ImgProcess.h"
int main() {
	// 提供应用程序的入口和一个命令行界面
	// 图源来自三个图像
	// 第一张，大图，读取raw图像的数据
	ImgProcess hugeImg("20180620-tianjin-2076x2816x3BIP", 2076, 2816, 1);
	// 第二张，中等图像，来自示例图像
	ImgProcess midImg("ik_beijing_p.bmp");
	// 最后一张，极小图像，仅用于演示自相关函数分析法
	ImgProcess minImg("ik_beijing_c_min.jpg");
	int cho = -1;
	while (cho != 0) {
		cout << "===================" << endl;
		cout << "======欢迎使用!======" << endl;
		cout << "=======Menu========" << endl;
		cout << "1.（必做1）图像点运算：灰度线性变换" << endl;
		cout << "2.（必做2）图像局部处理：高通滤波、低通滤波、中值滤波" << endl;
		cout << "3.（选做1）图像的几何处理：平移、缩放、旋转" << endl;
		cout << "4.（选做2）图像二值化：状态法及判断分析法" << endl;
		cout << "5.（选做3）纹理图像的自相关函数分析法" << endl;
		cout << "6.（选做4）伪彩色增强" << endl;
		cout << "7.（选做5）色彩平衡" << endl;
		cout << "0. 退出程序" << endl;
		cout << "你的选项是：";
		cin >> cho;
		// 清空屏幕
		system("cls");
		// 定义一些临时变量
		double k, b;
		int cho2, dr, dg, db;
		switch (cho) {
		case 0:
			exit(0);
		case 1:
			cout << "关闭图像窗口以运行下一项测试" << endl;
			cout << "======欢迎使用!======" << endl;
			cout << "=====灰度线性变换=====" << endl;
			cout << "输入线性变换函数的斜率（k）：";
			k = 0;
			cin >> k;
			cout << endl << "输入线性变换函数的截距（b）：";
			b = 0;
			cin >> b;
			hugeImg.grayLinear(k, b);
			cout << endl << "处理完毕！" << endl;
			break;
		case 2:
			cout << "关闭图像窗口以运行下一项测试" << endl;
			cout << "======欢迎使用!======" << endl;
			cout << "=====图像局部处理=====" << endl;
			cout << "选择高通滤波的方式（1.书本H1矩阵 2.书本H2矩阵（三阶拉普拉斯） 3.五阶拉普拉斯）：";
			cin >> cho2;
			hugeImg.highPass(cho2);
			hugeImg.lowPass();
			hugeImg.medPass();
			cout << endl << "处理完毕！" << endl;
			break;
		case 3:
			cout << "关闭图像窗口以运行下一项测试" << endl;
			cout << "======欢迎使用!======" << endl;
			cout << "=====图像几何处理=====" << endl;
			cout << "选择几何处理的方式(1.平移 2.缩放 3.旋转)：";
			cin >> cho2;
			switch (cho2)
			{
			case 1:
				cout << endl << "分两次，按顺序输入x和y方向偏离的像素值" << endl;
				int dx, dy;
				cin >> dx;
				cin >> dy;
				hugeImg.imgMove(dx, dy, 0);
				hugeImg.imgMove(dx, dy, 1);
				break;
			case 2:
				cout << endl << "分两次，按顺序输入x和y方向的缩放倍率" << endl;
				int rx, ry;
				cin >> rx;
				cin >> ry;
				hugeImg.imgZoom(rx, ry);
				break;
				cout << endl << "输入角度值的旋转角度（顺时针旋转）" << endl;
				double ang;
				hugeImg.imgRotate(ang);
				break;
			}
			cout << endl << "处理完毕！" << endl;
			break;
		case 4:
			cout << "关闭图像窗口以运行下一项测试" << endl;
			cout << "======欢迎使用!======" << endl;
			cout << "=====图像的二值化=====" << endl;
			midImg.imgBinOTSU();
			midImg.imgBinRatio();
			break;
		case 5:
			cout << "关闭图像窗口以运行下一项测试" << endl;
			cout << "======欢迎使用!======" << endl;
			cout << "====图像自相关分析====" << endl;
			minImg.imgAutoCorr();
			cout << endl << "处理完毕！" << endl;
			break;
		case 6:
			cout << "关闭图像窗口以运行下一项测试" << endl;
			cout << "======欢迎使用!======" << endl;
			cout << "====图像伪彩色增强====" << endl;
			hugeImg.bin2Color();
			cout << endl << "处理完毕！" << endl;
			break;
		case 7:
			cout << "关闭图像窗口以运行下一项测试" << endl;
			cout << "======欢迎使用!======" << endl;
			cout << "====图像的色彩平衡====" << endl;
			cout << "选择色彩平衡的方式(1.自定义 2.白平衡)：";
			cin >> cho2;
			switch (cho2)
			{
			case 1:
				cout << endl << "分三次，按顺序输入RGB改变的值" << endl;
				cin >> dr;
				cin >> dg;
				cin >> db;
				hugeImg.colorBalance(dr, dg, db);
				break;
			case 2:
				hugeImg.whiteBalance();
				break;
			}
			cout << endl << "处理完毕！" << endl;
			break;
		default:
			cout << "输入越界！请重新输入" << endl;
			system("pause");
		}
		cout << "关闭所有图像窗口以测试其他项目" << endl;
		system("pause");
		system("cls");
	}
	waitKey();
}