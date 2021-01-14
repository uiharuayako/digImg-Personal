#include "ImgProcess.h"

void ImgProcess::imgShowSave(Mat img, string title)
{
	showRawImg();
	namedWindow(title, 0);
	imshow(title, img);
	imwrite(title + ".bmp", img);
	waitKey();
}

ImgProcess::ImgProcess(string imgPath, int cols, int rows, int type)
{
	// 将参数读入类
	this->rawImgPath = imgPath;
	this->imCols = cols;
	this->imRows = rows;
	this->type = type;
	this->pixels = cols * rows * 3;
	// 文件读取操作
	FILE* fp;
	if (fopen_s(&fp,rawImgPath.c_str(),"rb") != 0) {
		cout<<"无法打开文件，请检查文件名"<<endl;
		waitKey(); // opencv中常用waitKey(n),n<=0表示一直
		exit(0);
	}
	auto* data = new uchar[pixels * sizeof(uchar)];// 预分配内存
	fread(data, sizeof(uchar), pixels, fp);// 将文件指针中内容读入内存
	fclose(fp);
	// 文件读取完毕，依照类型预处理图像
	Mat M(rows, cols, CV_8UC3, Scalar(0, 0, 0));
	unsigned char* ptr = M.data; // data是存数据的部分指针
	// 以下代码完成raw图像的读入，支持三种遥感影像格式的读入
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < 3; k++) {
				switch (type)
				{
				case 1:
					ptr[(i * cols + j) * 3 + k] = data[(i * cols + j) * 3 + (k + 1)];//BIP
					break;
				case 2:
					ptr[(i * cols + j) * 3 + k] = data[i * cols * 3 + (k + 3) * cols + j];//BIL
					break;
				case 3:
					ptr[(i * cols + j) * 3 + k] = data[(i + rows * k) * cols + j];//BSQ
					break;
				default:
					cout << "请指定正确的文件类型！" << endl;
				}
			}
		}
	}
	myImg = M;
	this->pSrc = myImg.data;
	this->channelNum = myImg.channels();
}

ImgProcess::ImgProcess(string imgPath)
{
	myImg = imread(imgPath);
	this->pSrc = myImg.data;
	this->channelNum = myImg.channels();
	// 将参数读入类
	this->rawImgPath = imgPath;
	this->imCols = myImg.cols;
	this->imRows = myImg.rows;
	this->type = myImg.type();
	this->pixels = imCols * imRows * 3;
}

void ImgProcess::showRawImg()
{
	// 测试用函数
	namedWindow("原始图像", 0);
	imshow("原始图像",myImg);
}

void ImgProcess::grayLinear(double k, double b)
{
	// 这个初始化，直接使用原图的大小和格式，非常方便
	Mat grayLineImg = Mat::zeros(myImg.size(), myImg.type());
	for (int row = 0; row < imRows; row++) {
		for (int col = 0; col < imCols; col++) {
			if (myImg.channels() == 3) {
				// 在opencv里，颜色是按照b，g，r排序的
				int b = myImg.at<Vec3b>(row, col)[0];
				int g = myImg.at<Vec3b>(row, col)[1];
				int r = myImg.at<Vec3b>(row, col)[2];
				// saturate_cast 处理指定数据类型的溢出
				// 保证了图像在进行灰度线性变换后不会产生色彩溢出
				// 以后的代码多次使用了这个方法
				// Vec3b表示这是个彩色图像
				grayLineImg.at<Vec3b>(row, col)[0] = saturate_cast<uchar>((k * b + b));
				grayLineImg.at<Vec3b>(row, col)[1] = saturate_cast<uchar>((k * g + b));
				grayLineImg.at<Vec3b>(row, col)[2] = saturate_cast<uchar>((k * r + b));
			}
			else if (myImg.channels() == 1) {
				// 对灰度图像进行处理
				int v = myImg.at<uchar>(row, col);
				grayLineImg.at<uchar>(row, col) = saturate_cast<uchar>(k * v + b);
			}
		}
	}
	imgShowSave(grayLineImg, "灰度线性变换");
}

cv::Mat ImgProcess::pass(vector<double> myTemplate)
{
	Mat passImg = Mat::zeros(myImg.size(), myImg.type());
	unsigned char* pDst = passImg.data;

	int tempSize = sqrt(myTemplate.size()); // 读取模板大小
	int calSize = (tempSize - 1) / 2; // 用于计算的模板大小
	//vector<int> myPixels; // 为模板涉及到的内存开辟一个vector数组
	auto* myPixels = new double(myTemplate.size()); // 为模板涉及到的内存开辟一个int数组
	int pixelNum = 0; // 当前处理的是第n个像素
	int totalpixel = 0; // 像素总灰度级
	// 五层嵌套循环，第一层，n个通道
	for (int i = 0; i < channelNum; i++) {
		// 第二，三层，限制边缘的行列不参加运算
		for (int j = calSize; j < imRows - calSize; j++) {
			for (int k = calSize; k < imCols - calSize; k++) {
				// 第四，五层，计算中间像素旁边的像素
#if 1
				for (int m = j - calSize; m <= j + calSize; m++) {
					for (int n = k - calSize; n <= k + calSize; n++) {
						myPixels[pixelNum] = myTemplate[pixelNum] * pSrc[((m)*imCols + (n)) * channelNum + i];
						pixelNum++;
					}
				}
 				totalpixel = 0;
 				for (int tmp = 0; tmp < pixelNum; tmp++) {
 					totalpixel = totalpixel + myPixels[tmp];
 				}
				// 限制灰度级使之不溢出
				pixelNum = 0;
#endif
				// stl的写法更加整洁美观，可以少定义很多变量。但是速度很慢，还是用数组吧
				// 以下代码亦可以运行，使用预处理器块注释掉
#if 0
				for (int m = j - calSize; m <= j + calSize; m++) {
					for (int n = k - calSize; n <= k + calSize; n++) {
						// 这句话是运算的核心，解释一下
						// 在此运用了vector，就是向vector数组末尾添加一个元素，这个元素是模板的第size个元素乘以相应位置的像素
						myPixels.push_back(myTemplate[myPixels.size()] * pSrc[((m)*imCols + (n)) * channelNum + i]);
					}
				}
				totalpixel = accumulate(myPixels.begin(), myPixels.end(), 0);// 累加各个像素
				myPixels.clear(); // 清空临时的像素数组
#endif
				// 限制灰度级使之不溢出
				pDst[(j * imCols + k) * channelNum + i] = saturate_cast<uchar>(totalpixel);
			}
		}
	}
	return passImg;
}

void ImgProcess::highPass(int type)
{
	switch (type)
	{
	case 1:
		imgShowSave(pass(hi1Template), "高通滤波");
		break;
	case 2:
		imgShowSave(pass(hi2Template), "高通滤波");
		break;
	case 3:
		imgShowSave(pass(hi3Template), "五阶拉普拉斯");
	}
}

void ImgProcess::lowPass()
{
	imgShowSave(pass(lowTemplate), "低通滤波");
}

void ImgProcess::medPass()
{
	Mat midImg = Mat::zeros(myImg.size(), myImg.type());
	unsigned char* pDst = midImg.data;

	uchar p[9] = { 0 };

	for (int k = 0; k < channelNum; k++) {
		for (int i = 1; i < imRows - 1; i++) {
			for (int j = 1; j < imCols - 1; j++) {
				// 找到目标像素点周围3*3的像素
				p[0] = pSrc[((i - 1) * imCols + (j - 1)) * channelNum + k];
				p[1] = pSrc[((i - 1) * imCols + j) * channelNum + k];
				p[2] = pSrc[((i - 1) * imCols + (j + 1)) * channelNum + k];
				p[3] = pSrc[(i * imCols + (j - 1)) * channelNum + k];
				p[4] = pSrc[(i * imCols + j) * channelNum + k];
				p[5] = pSrc[(i * imCols + (j + 1)) * channelNum + k];
				p[6] = pSrc[((i + 1) * imCols + (j - 1)) * channelNum + k];
				p[7] = pSrc[((i + 1) * imCols + j) * channelNum + k];
				p[8] = pSrc[((i + 1) * imCols + (j + 1)) * channelNum + k];
				// 使用std的sort排序一下
				sort(p, p + 9);
				pDst[(i * imCols + j) * channelNum + k] = p[5];
			}
		}
	}
	imgShowSave(midImg, "中值滤波");
}

void ImgProcess::imgMove(int dx, int dy, int type)
{
	Mat movedImg;
	Vec3b* p;
	int rows;
	int cols;
	switch (type) {
	case 0:
		// 不改变大小的写法
		rows = myImg.rows;
		cols = myImg.cols;
		movedImg.create(rows, cols, myImg.type());
		for (int i = 0; i < rows; i++)
		{
			p = movedImg.ptr<Vec3b>(i);
			for (int j = 0; j < cols; j++)
			{
				// 平移后坐标映射到原图像
				int x = j - dx;
				int y = i - dy;

				// 保证映射后的坐标在原图像范围内
				if (x >= 0 && y >= 0 && x < cols && y < rows)
					p[j] = myImg.ptr<Vec3b>(y)[x];
			}
		}
		imgShowSave(movedImg, "平移后大小不变");
		break;
	case 1:
		// 改变大小的写法
		rows = myImg.rows + abs(dy); //输出图像的大小
		cols = myImg.cols + abs(dx);
		movedImg.create(rows, cols, myImg.type());
		for (int i = 0; i < rows; i++)
		{
			p = movedImg.ptr<Vec3b>(i);
			for (int j = 0; j < cols; j++)
			{
				int x = j - dx;
				int y = i - dy;

				if (x >= 0 && y >= 0 && x < myImg.cols && y < myImg.rows)
					p[j] = myImg.ptr<Vec3b>(y)[x];
			}
		}
		imgShowSave(movedImg, "平移后改变大小");
		break;
	}
}

void ImgProcess::imgZoom(double xRate, double yRate)
{
	// 缩放图像，原理是构造一个3*3矩阵
	int rows = imRows * xRate;
	int cols = imCols * yRate;

	Mat zoomedImg = Mat::zeros(rows, cols, myImg.type());
	uchar* pDst = zoomedImg.data;

	// 构造缩放变换矩阵
	Mat T = (Mat_<double>(3, 3) << xRate, 0, 0, 0, yRate, 0, 0, 0, 1);
	Mat T_inv = T.inv();//求逆

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//第i行j列的像素坐标(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//原图像的横坐标，对应图像的列数
			double v = src_uv.at<double>(0, 1);//原图像的纵坐标，对应图像的行数

			// 双线性插值法插值运算不存在的像素点
			if (u >= 0 && v >= 0 && u <= imCols - 1 && v <= imRows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //与映射到原图坐标相邻的四个像素点的坐标
				double dv = v - top; // dv为坐标 行 的小数部分(坐标偏差)
				double du = u - left; // du为坐标 列 的小数部分(坐标偏差)

				for (int k = 0; k < channelNum; k++) {
					pDst[(i * cols + j) * channelNum + k] = (1 - dv) * (1 - du) * pSrc[(top * imCols + left) * channelNum + k] + (1 - dv) * du * pSrc[(top * imCols + right) * channelNum + k] + dv * (1 - du) * pSrc[(bottom * imCols + left) * channelNum + k] + dv * du * pSrc[(bottom * imCols + right) * channelNum + k];
				}
			}
		}
	}
	imgShowSave(zoomedImg, "缩放后图像");
}

void ImgProcess::imgRotate(double angle)
{
	angle = angle * CV_PI / 180;
	int rows = round(fabs(imRows * cos(angle)) + fabs(imCols * sin(angle)));
	int cols = round(fabs(imCols * cos(angle)) + fabs(imRows * sin(angle)));

	Mat rotImg = Mat::zeros(rows, cols, myImg.type());
	uchar* pDst = rotImg.data;

	//构造旋转变换矩阵
	Mat T1 = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.5 * imCols, 0.5 * imRows, 1.0);
	Mat T2 = (Mat_<double>(3, 3) << cos(angle), -sin(angle), 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0, 1.0);
	double t3[3][3] = { { 1.0, 0.0, 0.0 },{ 0.0, -1.0, 0.0 },{ 0.5 * rotImg.cols, 0.5 * rotImg.rows ,1.0 } }; // 将数学笛卡尔坐标映射到旋转后的图像坐标
	Mat T3 = Mat(3.0, 3.0, CV_64FC1, t3);

	Mat T = T1 * T2 * T3;
	Mat T_inv = T.inv();//求逆

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//第i行j列的像素坐标(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//原图像的横坐标，对应图像的列数
			double v = src_uv.at<double>(0, 1);//原图像的纵坐标，对应图像的行数

			//双线性插值法
			if (u >= 0 && v >= 0 && u <= imCols - 1 && v <= imRows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //与映射到原图坐标相邻的四个像素点的坐标
				double dv = v - top; //dv为坐标 行 的小数部分(坐标偏差)
				double du = u - left; //du为坐标 列 的小数部分(坐标偏差)

				for (int k = 0; k < channelNum; k++) {
					pDst[(i * cols + j) * channelNum + k] = (1 - dv) * (1 - du) * pSrc[(top * imCols + left) * channelNum + k] + (1 - dv) * du * pSrc[(top * imCols + right) * channelNum + k] + dv * (1 - du) * pSrc[(bottom * imCols + left) * channelNum + k] + dv * du * pSrc[(bottom * imCols + right) * channelNum + k];
				}
			}


		}
	}

	imgShowSave(rotImg, "图像旋转");
}

void ImgProcess::imgBinRatio()
{
	// 首先搞个灰度图像，直接二值化计算量太太太大了
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	int nEISize = binImg.elemSize();// 获取每个像素的字节数
	int G = pow(2, double(8 * nEISize));// 灰度级数
	auto* imHist = new int[G];// 分配内存，用于储存灰度直方图数组
	// 依据定义，数组的每一项最低为0
	for (int i = 0; i < G; i++) {
		imHist[i] = 0;
	}
	// 计算灰度直方图
	for (int i = 0; i < imRows; i++) {
		for (int j = 0; j < imCols; j++) {
			int H = binImg.at<uchar>(i, j);
			imHist[H]++;
		}
	}
	// 阈值T
	// 通过使组内方差与组间方差之比最大来确定阈值
	double ratio = 0;
	int T = 0;
	//两组像素的总数量
	double tot1 = 0;
	double tot2 = 0;
	//两组像素的灰度平均值以及整幅图像的灰度平均值
	double med1 = 0;
	double med2 = 0;
	double med = 0;

	double σ12 = 0;//第一组像素的方差
	double σ22 = 0;//第二组像素的方差
	double σw2 = 0;//组内方差
	double σb2 = 0;//组间方差

	for (int i = 0; i < G; i++) {
		for (int m = i; m < G; m++) {
			tot2 += imHist[m];
		}
		tot1 = imCols * imRows - tot2;
		for (int n = i; n < G; n++) {
			med2 += (double(imHist[n]) / tot2) * n;
		}
		for (int k = 0; k < i; k++) {
			med1 += (double(imHist[k]) / tot1) * k;
		}
		//整幅图像的灰度平均值计算
		med = (med1 * tot1 + med2 * tot2) / (tot1 + tot2);
		//两组的方差以及组内方差和组间方差计算
		for (int p = i; p < G; p++) {
			σ22 += (double(imHist[p]) / tot2) * pow((p - med2), 2);
		}
		for (int q = 0; q < i; q++) {
			σ12 += (double(imHist[q]) / tot1) * pow((q - med1), 2);
		}
		σw2 = tot1 * σ12 + tot2 * σ22;
		σb2 = tot1 * tot2 * pow((med1 - med2), 2);

		if ((σb2 / σw2) > ratio) {
			ratio = σb2 / σw2;
			T = i;//阈值T计算
		}
		//重新赋值为0
		tot1 = 0;
		tot2 = 0;
		med1 = 0;
		med2 = 0;
		σ12 = 0;
		σ22 = 0;
		σw2 = 0;
		σb2 = 0;
	}
	//根据阈值进行二值化处理
	for (int i = 0; i < imRows; i++) {
		for (int j = 0; j < imCols; j++) {
			if (binImg.at<uchar>(i, j) < T) {
				binImg.at<uchar>(i, j) = 0;
			}
			else if (binImg.at<uchar>(i, j) >= T) {
				binImg.at<uchar>(i, j) = G - 1;
			}
		}
	}
	imgShowSave(binImg, "比值法二值化");
}

void ImgProcess::imgBinOTSU()
{
	// 首先搞个灰度图像，直接二值化计算量太太太大了
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	int nEISize = binImg.elemSize();// 获取每个像素的字节数
	int G = pow(2, double(8 * nEISize));// 灰度级数
	auto* imHist = new int[G];// 分配内存，用于储存灰度直方图数组
	auto* ratHist = new double[G];// 分配内存，用于储存各个灰度级的概率
	// 依据定义，数组的每一项最低为0
	for (int i = 0; i < G; i++) {
		imHist[i] = 0;
	}
	// 计算灰度直方图
	for (int i = 0; i < imRows; i++) {
		for (int j = 0; j < imCols; j++) {
			int H = binImg.at<uchar>(i, j);
			imHist[H]++;
		}
	}
	// 阈值T
	// 通过otsu算法来确定阈值
	int i, j;
	int temp;
	//第一类均值，第二类均值，全局均值，mk=p1*m1, 第一类概率，第二类概率
	double m1, m2, mG, mk, p1, p2;

	double cov;
	double maxcov = 0.0;
	int T = 0;

	//计算每个灰度级占图像的概率
	for (i = 0; i < G; ++i)
		ratHist[i] = (double)imHist[i] / (double)(imRows * imCols);
	//计算平均灰度值
	mG = 0.0;
	for (i = 0; i < G; ++i)
		mG += i * ratHist[i];
	//统计前景和背景的平均灰度值，并计算类间方差
	for (i = 0; i < G; ++i)
	{
		m1 = 0.0; m2 = 0.0; mk = 0.0; p1 = 0.0; p2 = 0.0;
		for (j = 0; j < i; ++j) {
			p1 += ratHist[j];
			mk += j * ratHist[j];
		}
		m1 = mk / p1;  //mk=p1*m1,是一个中间值
		p2 = 1 - p1;  //p1+p2=1;
		m2 = (mG - mk) / p2;  //mG=p1*m1+p2*m2;
		//计算类间方差
		cov = p1 * p2 * (m1 - m2) * (m1 - m2);
		if (cov > maxcov) {
			maxcov = cov;
			T = i;
		}
	}
	//根据阈值进行二值化处理
	for (int i = 0; i < imRows; i++) {
		for (int j = 0; j < imCols; j++) {
			if (binImg.at<uchar>(i, j) < T) {
				binImg.at<uchar>(i, j) = 0;
			}
			else if (binImg.at<uchar>(i, j) >= T) {
				binImg.at<uchar>(i, j) = G - 1;
			}
		}
	}
	imgShowSave(binImg, "OTSU法二值化");
}

void ImgProcess::imgAutoCorr()
{
	// 首先还是搞个灰度图像
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	int i, j; // i，j嵌套循环用了两次，最好前面就定义一下
	// 根据自相关函数那个很复杂的式子来写，自相关函数的分母是固定的，其分母设为deno
	int deno = 0;
	for (i = 0; i < imRows; i++) {
		for (j = 0; j < imCols; j++) {
			// 两次循环嵌套，遍历原来的图像
			// 计算像素值平方和
			deno = deno + (int)binImg.at<uchar>(i, j) * (int)binImg.at<uchar>(i, j);
		}
	}
	// 然后创建一个目标图像
	int nume = 0;// 搞一个分子，分子每次都在变
	Mat autoCorrImg;
	autoCorrImg.create(myImg.size(), binImg.type());
	for(i = 0; i < imRows; i++) {
		for (j = 0; j < imCols; j++) {
			// 两次循环嵌套，遍历新的图像
			for (int m = 0; m < imRows; m++) {
				for (int n = 0; n < imCols; n++) {
					// 两次循环嵌套，就是分子
					// 这里后面的m+i，n+j就是书上的i+x，j+y
					if (m + i > imRows - 1 || n + j > imCols - 1) {
						nume = 0;
					}
					else
					{
						// 这行代码实现分子的累加
						nume = nume + (int)binImg.at<uchar>(m, n) * (int)binImg.at<uchar>(m + i, n + j);
					}
				}
			}
			// 分子计算完毕
			// 对当前像素赋值
			autoCorrImg.at<uchar>(i, j) = saturate_cast<uchar>(nume / deno);
			// 分子归0，等待下次计算
			nume = 0;
		}
	}
	imgShowSave(autoCorrImg, "自相关函数图像");
}

void ImgProcess::bin2Color()
{
	// 整个灰度图像
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	// 设置三种颜色
	Mat R = binImg.clone();
	Mat G = binImg.clone();
	Mat B = binImg.clone();
	// 开始进行空间域彩色合成，使用的是分段线性函数
	for (int i = 0; i < imRows; i++)
	{
		for (int j = 0; j < imCols; j++)
		{
			int current = binImg.at<uchar>(i, j);
			if (0 < current && current < 64)
			{
				R.at<uchar>(i, j) = 0;
				G.at<uchar>(i, j) = 4 * current;
				B.at<uchar>(i, j) = 255;
			}
			else if (64 <= current && current <= 128)
			{
				R.at<uchar>(i, j) = 0;
				G.at<uchar>(i, j) = 255;
				B.at<uchar>(i, j) = -4 * (current - 64) + 255;
			}
			else if (128 < current && current < 192)
			{
				R.at<uchar>(i, j) = 4 * (current - 128);
				G.at<uchar>(i, j) = 255;
				B.at<uchar>(i, j) = 0;
			}
			else
			{
				R.at<uchar>(i, j) = 255;
				G.at<uchar>(i, j) = -4 * (current - 255);
				B.at<uchar>(i, j) = 0;
			}


		}
	}

	Mat channels[3];  //定义对象数组，分别存储三个通道
	Mat fakeColoredImg;      //融合三个通道，存储在一个Mat里
	channels[0] = B;
	channels[1] = G;
	channels[2] = R;
	merge(channels, 3, fakeColoredImg);
	imgShowSave(fakeColoredImg, "伪彩色图像");
}

void ImgProcess::whiteBalance()
{
	Mat dst;
	dst.create(myImg.size(), myImg.type());
	int HistRGB[767] = { 0 };//设置一个数组
	int MaxVal = 0;
	//统计R+G+B，得到R，G，B中的最大值MaxVal
	for (int i = 0; i < myImg.rows; i++)
	{
		for (int j = 0; j < myImg.cols; j++)
		{
			MaxVal = max(MaxVal, (int)myImg.at<Vec3b>(i, j)[0]);
			MaxVal = max(MaxVal, (int)myImg.at<Vec3b>(i, j)[1]);
			MaxVal = max(MaxVal, (int)myImg.at<Vec3b>(i, j)[2]);
			int sum = myImg.at<Vec3b>(i, j)[0] + myImg.at<Vec3b>(i, j)[1] + myImg.at<Vec3b>(i, j)[2];
			HistRGB[sum]++;
		}
	}
	//计算R+G+B的数量超过像素总数的ratio的像素值，计算出阈值Threshold
	int Threshold = 0;
	int sum = 0;
	float ratio = 0.1;
	for (int i = 766; i >= 0; i--) {
		sum += HistRGB[i];
		if (sum > myImg.rows * myImg.cols * ratio) {
			Threshold = i;
			break;
		}
	}
	//计算R+G+B大于阈值的所有点的均值
	int AvgB = 0;
	int AvgG = 0;
	int AvgR = 0;
	int cnt = 0;
	for (int i = 0; i < myImg.rows; i++) {
		for (int j = 0; j < myImg.cols; j++) {
			int sumP = myImg.at<Vec3b>(i, j)[0] + myImg.at<Vec3b>(i, j)[1] + myImg.at<Vec3b>(i, j)[2];
			if (sumP > Threshold) {
				AvgB += myImg.at<Vec3b>(i, j)[0];
				AvgG += myImg.at<Vec3b>(i, j)[1];
				AvgR += myImg.at<Vec3b>(i, j)[2];
				cnt++;
			}
		}
	}
	//得到均值
	AvgB /= cnt;
	AvgG /= cnt;
	AvgR /= cnt;
	//量化0-255，得到新矩阵dst
	for (int i = 0; i < myImg.rows; i++) {
		for (int j = 0; j < myImg.cols; j++) {
			int Blue = myImg.at<Vec3b>(i, j)[0] * MaxVal / AvgB;
			int Green = myImg.at<Vec3b>(i, j)[1] * MaxVal / AvgG;
			int Red = myImg.at<Vec3b>(i, j)[2] * MaxVal / AvgR;
			if (Red > 255) {
				Red = 255;
			}
			else if (Red < 0) {
				Red = 0;
			}
			if (Green > 255) {
				Green = 255;
			}
			else if (Green < 0) {
				Green = 0;
			}
			if (Blue > 255) {
				Blue = 255;
			}
			else if (Blue < 0) {
				Blue = 0;
			}
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	imgShowSave(dst, "色彩平衡后");
}

void ImgProcess::colorBalance(int deltaR, int deltaG, int deltaB)
{
	Mat cbImg; // 用于储存色彩平衡后的图像
	cbImg.create(myImg.size(), myImg.type());
	// 初始，全部赋0
	cbImg = cv::Scalar::all(0);

	for (int i = 0; i < imRows; i++)
	{
		auto* src = (uchar*)(myImg.data + myImg.step * i);
		auto* dst = (uchar*)cbImg.data + cbImg.step * i;
		for (int j = 0; j < imCols; j++)
		{
			// 注意，opencv的RGB是反的
			dst[j * channelNum] = saturate_cast<uchar>(src[j * channelNum] + deltaB);
			dst[j * channelNum + 1] = saturate_cast<uchar>(src[j * channelNum + 1] + deltaG);
			dst[j * channelNum + 2] = saturate_cast<uchar>(src[j * channelNum + 2] + deltaR);
		}
	}
	imgShowSave(cbImg, "自定义色彩平衡后");
}
