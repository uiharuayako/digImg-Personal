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
	// ������������
	this->rawImgPath = imgPath;
	this->imCols = cols;
	this->imRows = rows;
	this->type = type;
	this->pixels = cols * rows * 3;
	// �ļ���ȡ����
	FILE* fp;
	if (fopen_s(&fp,rawImgPath.c_str(),"rb") != 0) {
		cout<<"�޷����ļ��������ļ���"<<endl;
		waitKey(); // opencv�г���waitKey(n),n<=0��ʾһֱ
		exit(0);
	}
	auto* data = new uchar[pixels * sizeof(uchar)];// Ԥ�����ڴ�
	fread(data, sizeof(uchar), pixels, fp);// ���ļ�ָ�������ݶ����ڴ�
	fclose(fp);
	// �ļ���ȡ��ϣ���������Ԥ����ͼ��
	Mat M(rows, cols, CV_8UC3, Scalar(0, 0, 0));
	unsigned char* ptr = M.data; // data�Ǵ����ݵĲ���ָ��
	// ���´������rawͼ��Ķ��룬֧������ң��Ӱ���ʽ�Ķ���
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
					cout << "��ָ����ȷ���ļ����ͣ�" << endl;
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
	// ������������
	this->rawImgPath = imgPath;
	this->imCols = myImg.cols;
	this->imRows = myImg.rows;
	this->type = myImg.type();
	this->pixels = imCols * imRows * 3;
}

void ImgProcess::showRawImg()
{
	// �����ú���
	namedWindow("ԭʼͼ��", 0);
	imshow("ԭʼͼ��",myImg);
}

void ImgProcess::grayLinear(double k, double b)
{
	// �����ʼ����ֱ��ʹ��ԭͼ�Ĵ�С�͸�ʽ���ǳ�����
	Mat grayLineImg = Mat::zeros(myImg.size(), myImg.type());
	for (int row = 0; row < imRows; row++) {
		for (int col = 0; col < imCols; col++) {
			if (myImg.channels() == 3) {
				// ��opencv���ɫ�ǰ���b��g��r�����
				int b = myImg.at<Vec3b>(row, col)[0];
				int g = myImg.at<Vec3b>(row, col)[1];
				int r = myImg.at<Vec3b>(row, col)[2];
				// saturate_cast ����ָ���������͵����
				// ��֤��ͼ���ڽ��лҶ����Ա任�󲻻����ɫ�����
				// �Ժ�Ĵ�����ʹ�����������
				// Vec3b��ʾ���Ǹ���ɫͼ��
				grayLineImg.at<Vec3b>(row, col)[0] = saturate_cast<uchar>((k * b + b));
				grayLineImg.at<Vec3b>(row, col)[1] = saturate_cast<uchar>((k * g + b));
				grayLineImg.at<Vec3b>(row, col)[2] = saturate_cast<uchar>((k * r + b));
			}
			else if (myImg.channels() == 1) {
				// �ԻҶ�ͼ����д���
				int v = myImg.at<uchar>(row, col);
				grayLineImg.at<uchar>(row, col) = saturate_cast<uchar>(k * v + b);
			}
		}
	}
	imgShowSave(grayLineImg, "�Ҷ����Ա任");
}

cv::Mat ImgProcess::pass(vector<double> myTemplate)
{
	Mat passImg = Mat::zeros(myImg.size(), myImg.type());
	unsigned char* pDst = passImg.data;

	int tempSize = sqrt(myTemplate.size()); // ��ȡģ���С
	int calSize = (tempSize - 1) / 2; // ���ڼ����ģ���С
	//vector<int> myPixels; // Ϊģ���漰�����ڴ濪��һ��vector����
	auto* myPixels = new double(myTemplate.size()); // Ϊģ���漰�����ڴ濪��һ��int����
	int pixelNum = 0; // ��ǰ������ǵ�n������
	int totalpixel = 0; // �����ܻҶȼ�
	// ���Ƕ��ѭ������һ�㣬n��ͨ��
	for (int i = 0; i < channelNum; i++) {
		// �ڶ������㣬���Ʊ�Ե�����в��μ�����
		for (int j = calSize; j < imRows - calSize; j++) {
			for (int k = calSize; k < imCols - calSize; k++) {
				// ���ģ���㣬�����м������Աߵ�����
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
				// ���ƻҶȼ�ʹ֮�����
				pixelNum = 0;
#endif
				// stl��д�������������ۣ������ٶ���ܶ�����������ٶȺ����������������
				// ���´�����������У�ʹ��Ԥ��������ע�͵�
#if 0
				for (int m = j - calSize; m <= j + calSize; m++) {
					for (int n = k - calSize; n <= k + calSize; n++) {
						// ��仰������ĺ��ģ�����һ��
						// �ڴ�������vector��������vector����ĩβ���һ��Ԫ�أ����Ԫ����ģ��ĵ�size��Ԫ�س�����Ӧλ�õ�����
						myPixels.push_back(myTemplate[myPixels.size()] * pSrc[((m)*imCols + (n)) * channelNum + i]);
					}
				}
				totalpixel = accumulate(myPixels.begin(), myPixels.end(), 0);// �ۼӸ�������
				myPixels.clear(); // �����ʱ����������
#endif
				// ���ƻҶȼ�ʹ֮�����
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
		imgShowSave(pass(hi1Template), "��ͨ�˲�");
		break;
	case 2:
		imgShowSave(pass(hi2Template), "��ͨ�˲�");
		break;
	case 3:
		imgShowSave(pass(hi3Template), "���������˹");
	}
}

void ImgProcess::lowPass()
{
	imgShowSave(pass(lowTemplate), "��ͨ�˲�");
}

void ImgProcess::medPass()
{
	Mat midImg = Mat::zeros(myImg.size(), myImg.type());
	unsigned char* pDst = midImg.data;

	uchar p[9] = { 0 };

	for (int k = 0; k < channelNum; k++) {
		for (int i = 1; i < imRows - 1; i++) {
			for (int j = 1; j < imCols - 1; j++) {
				// �ҵ�Ŀ�����ص���Χ3*3������
				p[0] = pSrc[((i - 1) * imCols + (j - 1)) * channelNum + k];
				p[1] = pSrc[((i - 1) * imCols + j) * channelNum + k];
				p[2] = pSrc[((i - 1) * imCols + (j + 1)) * channelNum + k];
				p[3] = pSrc[(i * imCols + (j - 1)) * channelNum + k];
				p[4] = pSrc[(i * imCols + j) * channelNum + k];
				p[5] = pSrc[(i * imCols + (j + 1)) * channelNum + k];
				p[6] = pSrc[((i + 1) * imCols + (j - 1)) * channelNum + k];
				p[7] = pSrc[((i + 1) * imCols + j) * channelNum + k];
				p[8] = pSrc[((i + 1) * imCols + (j + 1)) * channelNum + k];
				// ʹ��std��sort����һ��
				sort(p, p + 9);
				pDst[(i * imCols + j) * channelNum + k] = p[5];
			}
		}
	}
	imgShowSave(midImg, "��ֵ�˲�");
}

void ImgProcess::imgMove(int dx, int dy, int type)
{
	Mat movedImg;
	Vec3b* p;
	int rows;
	int cols;
	switch (type) {
	case 0:
		// ���ı��С��д��
		rows = myImg.rows;
		cols = myImg.cols;
		movedImg.create(rows, cols, myImg.type());
		for (int i = 0; i < rows; i++)
		{
			p = movedImg.ptr<Vec3b>(i);
			for (int j = 0; j < cols; j++)
			{
				// ƽ�ƺ�����ӳ�䵽ԭͼ��
				int x = j - dx;
				int y = i - dy;

				// ��֤ӳ����������ԭͼ��Χ��
				if (x >= 0 && y >= 0 && x < cols && y < rows)
					p[j] = myImg.ptr<Vec3b>(y)[x];
			}
		}
		imgShowSave(movedImg, "ƽ�ƺ��С����");
		break;
	case 1:
		// �ı��С��д��
		rows = myImg.rows + abs(dy); //���ͼ��Ĵ�С
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
		imgShowSave(movedImg, "ƽ�ƺ�ı��С");
		break;
	}
}

void ImgProcess::imgZoom(double xRate, double yRate)
{
	// ����ͼ��ԭ���ǹ���һ��3*3����
	int rows = imRows * xRate;
	int cols = imCols * yRate;

	Mat zoomedImg = Mat::zeros(rows, cols, myImg.type());
	uchar* pDst = zoomedImg.data;

	// �������ű任����
	Mat T = (Mat_<double>(3, 3) << xRate, 0, 0, 0, yRate, 0, 0, 0, 1);
	Mat T_inv = T.inv();//����

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//��i��j�е���������(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//ԭͼ��ĺ����꣬��Ӧͼ�������
			double v = src_uv.at<double>(0, 1);//ԭͼ��������꣬��Ӧͼ�������

			// ˫���Բ�ֵ����ֵ���㲻���ڵ����ص�
			if (u >= 0 && v >= 0 && u <= imCols - 1 && v <= imRows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double dv = v - top; // dvΪ���� �� ��С������(����ƫ��)
				double du = u - left; // duΪ���� �� ��С������(����ƫ��)

				for (int k = 0; k < channelNum; k++) {
					pDst[(i * cols + j) * channelNum + k] = (1 - dv) * (1 - du) * pSrc[(top * imCols + left) * channelNum + k] + (1 - dv) * du * pSrc[(top * imCols + right) * channelNum + k] + dv * (1 - du) * pSrc[(bottom * imCols + left) * channelNum + k] + dv * du * pSrc[(bottom * imCols + right) * channelNum + k];
				}
			}
		}
	}
	imgShowSave(zoomedImg, "���ź�ͼ��");
}

void ImgProcess::imgRotate(double angle)
{
	angle = angle * CV_PI / 180;
	int rows = round(fabs(imRows * cos(angle)) + fabs(imCols * sin(angle)));
	int cols = round(fabs(imCols * cos(angle)) + fabs(imRows * sin(angle)));

	Mat rotImg = Mat::zeros(rows, cols, myImg.type());
	uchar* pDst = rotImg.data;

	//������ת�任����
	Mat T1 = (Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, -0.5 * imCols, 0.5 * imRows, 1.0);
	Mat T2 = (Mat_<double>(3, 3) << cos(angle), -sin(angle), 0.0, sin(angle), cos(angle), 0.0, 0.0, 0.0, 1.0);
	double t3[3][3] = { { 1.0, 0.0, 0.0 },{ 0.0, -1.0, 0.0 },{ 0.5 * rotImg.cols, 0.5 * rotImg.rows ,1.0 } }; // ����ѧ�ѿ�������ӳ�䵽��ת���ͼ������
	Mat T3 = Mat(3.0, 3.0, CV_64FC1, t3);

	Mat T = T1 * T2 * T3;
	Mat T_inv = T.inv();//����

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			Mat dst_xy = (Mat_<double>(1, 3) << j, i, 1);//��i��j�е���������(j,i)
			Mat src_uv = dst_xy * T_inv;

			double u = src_uv.at<double>(0, 0);//ԭͼ��ĺ����꣬��Ӧͼ�������
			double v = src_uv.at<double>(0, 1);//ԭͼ��������꣬��Ӧͼ�������

			//˫���Բ�ֵ��
			if (u >= 0 && v >= 0 && u <= imCols - 1 && v <= imRows - 1) {
				int top = floor(v), bottom = ceil(v), left = floor(u), right = ceil(u); //��ӳ�䵽ԭͼ�������ڵ��ĸ����ص������
				double dv = v - top; //dvΪ���� �� ��С������(����ƫ��)
				double du = u - left; //duΪ���� �� ��С������(����ƫ��)

				for (int k = 0; k < channelNum; k++) {
					pDst[(i * cols + j) * channelNum + k] = (1 - dv) * (1 - du) * pSrc[(top * imCols + left) * channelNum + k] + (1 - dv) * du * pSrc[(top * imCols + right) * channelNum + k] + dv * (1 - du) * pSrc[(bottom * imCols + left) * channelNum + k] + dv * du * pSrc[(bottom * imCols + right) * channelNum + k];
				}
			}


		}
	}

	imgShowSave(rotImg, "ͼ����ת");
}

void ImgProcess::imgBinRatio()
{
	// ���ȸ���Ҷ�ͼ��ֱ�Ӷ�ֵ��������̫̫̫����
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	int nEISize = binImg.elemSize();// ��ȡÿ�����ص��ֽ���
	int G = pow(2, double(8 * nEISize));// �Ҷȼ���
	auto* imHist = new int[G];// �����ڴ棬���ڴ���Ҷ�ֱ��ͼ����
	// ���ݶ��壬�����ÿһ�����Ϊ0
	for (int i = 0; i < G; i++) {
		imHist[i] = 0;
	}
	// ����Ҷ�ֱ��ͼ
	for (int i = 0; i < imRows; i++) {
		for (int j = 0; j < imCols; j++) {
			int H = binImg.at<uchar>(i, j);
			imHist[H]++;
		}
	}
	// ��ֵT
	// ͨ��ʹ���ڷ�������䷽��֮�������ȷ����ֵ
	double ratio = 0;
	int T = 0;
	//�������ص�������
	double tot1 = 0;
	double tot2 = 0;
	//�������صĻҶ�ƽ��ֵ�Լ�����ͼ��ĻҶ�ƽ��ֵ
	double med1 = 0;
	double med2 = 0;
	double med = 0;

	double ��12 = 0;//��һ�����صķ���
	double ��22 = 0;//�ڶ������صķ���
	double ��w2 = 0;//���ڷ���
	double ��b2 = 0;//��䷽��

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
		//����ͼ��ĻҶ�ƽ��ֵ����
		med = (med1 * tot1 + med2 * tot2) / (tot1 + tot2);
		//����ķ����Լ����ڷ������䷽�����
		for (int p = i; p < G; p++) {
			��22 += (double(imHist[p]) / tot2) * pow((p - med2), 2);
		}
		for (int q = 0; q < i; q++) {
			��12 += (double(imHist[q]) / tot1) * pow((q - med1), 2);
		}
		��w2 = tot1 * ��12 + tot2 * ��22;
		��b2 = tot1 * tot2 * pow((med1 - med2), 2);

		if ((��b2 / ��w2) > ratio) {
			ratio = ��b2 / ��w2;
			T = i;//��ֵT����
		}
		//���¸�ֵΪ0
		tot1 = 0;
		tot2 = 0;
		med1 = 0;
		med2 = 0;
		��12 = 0;
		��22 = 0;
		��w2 = 0;
		��b2 = 0;
	}
	//������ֵ���ж�ֵ������
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
	imgShowSave(binImg, "��ֵ����ֵ��");
}

void ImgProcess::imgBinOTSU()
{
	// ���ȸ���Ҷ�ͼ��ֱ�Ӷ�ֵ��������̫̫̫����
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	int nEISize = binImg.elemSize();// ��ȡÿ�����ص��ֽ���
	int G = pow(2, double(8 * nEISize));// �Ҷȼ���
	auto* imHist = new int[G];// �����ڴ棬���ڴ���Ҷ�ֱ��ͼ����
	auto* ratHist = new double[G];// �����ڴ棬���ڴ�������Ҷȼ��ĸ���
	// ���ݶ��壬�����ÿһ�����Ϊ0
	for (int i = 0; i < G; i++) {
		imHist[i] = 0;
	}
	// ����Ҷ�ֱ��ͼ
	for (int i = 0; i < imRows; i++) {
		for (int j = 0; j < imCols; j++) {
			int H = binImg.at<uchar>(i, j);
			imHist[H]++;
		}
	}
	// ��ֵT
	// ͨ��otsu�㷨��ȷ����ֵ
	int i, j;
	int temp;
	//��һ���ֵ���ڶ����ֵ��ȫ�־�ֵ��mk=p1*m1, ��һ����ʣ��ڶ������
	double m1, m2, mG, mk, p1, p2;

	double cov;
	double maxcov = 0.0;
	int T = 0;

	//����ÿ���Ҷȼ�ռͼ��ĸ���
	for (i = 0; i < G; ++i)
		ratHist[i] = (double)imHist[i] / (double)(imRows * imCols);
	//����ƽ���Ҷ�ֵ
	mG = 0.0;
	for (i = 0; i < G; ++i)
		mG += i * ratHist[i];
	//ͳ��ǰ���ͱ�����ƽ���Ҷ�ֵ����������䷽��
	for (i = 0; i < G; ++i)
	{
		m1 = 0.0; m2 = 0.0; mk = 0.0; p1 = 0.0; p2 = 0.0;
		for (j = 0; j < i; ++j) {
			p1 += ratHist[j];
			mk += j * ratHist[j];
		}
		m1 = mk / p1;  //mk=p1*m1,��һ���м�ֵ
		p2 = 1 - p1;  //p1+p2=1;
		m2 = (mG - mk) / p2;  //mG=p1*m1+p2*m2;
		//������䷽��
		cov = p1 * p2 * (m1 - m2) * (m1 - m2);
		if (cov > maxcov) {
			maxcov = cov;
			T = i;
		}
	}
	//������ֵ���ж�ֵ������
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
	imgShowSave(binImg, "OTSU����ֵ��");
}

void ImgProcess::imgAutoCorr()
{
	// ���Ȼ��Ǹ���Ҷ�ͼ��
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	int i, j; // i��jǶ��ѭ���������Σ����ǰ��Ͷ���һ��
	// ��������غ����Ǹ��ܸ��ӵ�ʽ����д������غ����ķ�ĸ�ǹ̶��ģ����ĸ��Ϊdeno
	int deno = 0;
	for (i = 0; i < imRows; i++) {
		for (j = 0; j < imCols; j++) {
			// ����ѭ��Ƕ�ף�����ԭ����ͼ��
			// ��������ֵƽ����
			deno = deno + (int)binImg.at<uchar>(i, j) * (int)binImg.at<uchar>(i, j);
		}
	}
	// Ȼ�󴴽�һ��Ŀ��ͼ��
	int nume = 0;// ��һ�����ӣ�����ÿ�ζ��ڱ�
	Mat autoCorrImg;
	autoCorrImg.create(myImg.size(), binImg.type());
	for(i = 0; i < imRows; i++) {
		for (j = 0; j < imCols; j++) {
			// ����ѭ��Ƕ�ף������µ�ͼ��
			for (int m = 0; m < imRows; m++) {
				for (int n = 0; n < imCols; n++) {
					// ����ѭ��Ƕ�ף����Ƿ���
					// ��������m+i��n+j�������ϵ�i+x��j+y
					if (m + i > imRows - 1 || n + j > imCols - 1) {
						nume = 0;
					}
					else
					{
						// ���д���ʵ�ַ��ӵ��ۼ�
						nume = nume + (int)binImg.at<uchar>(m, n) * (int)binImg.at<uchar>(m + i, n + j);
					}
				}
			}
			// ���Ӽ������
			// �Ե�ǰ���ظ�ֵ
			autoCorrImg.at<uchar>(i, j) = saturate_cast<uchar>(nume / deno);
			// ���ӹ�0���ȴ��´μ���
			nume = 0;
		}
	}
	imgShowSave(autoCorrImg, "����غ���ͼ��");
}

void ImgProcess::bin2Color()
{
	// �����Ҷ�ͼ��
	imwrite("picPre.bmp", myImg);
	Mat binImg = imread("picPre.bmp", IMREAD_GRAYSCALE);
	// ����������ɫ
	Mat R = binImg.clone();
	Mat G = binImg.clone();
	Mat B = binImg.clone();
	// ��ʼ���пռ����ɫ�ϳɣ�ʹ�õ��Ƿֶ����Ժ���
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

	Mat channels[3];  //����������飬�ֱ�洢����ͨ��
	Mat fakeColoredImg;      //�ں�����ͨ�����洢��һ��Mat��
	channels[0] = B;
	channels[1] = G;
	channels[2] = R;
	merge(channels, 3, fakeColoredImg);
	imgShowSave(fakeColoredImg, "α��ɫͼ��");
}

void ImgProcess::whiteBalance()
{
	Mat dst;
	dst.create(myImg.size(), myImg.type());
	int HistRGB[767] = { 0 };//����һ������
	int MaxVal = 0;
	//ͳ��R+G+B���õ�R��G��B�е����ֵMaxVal
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
	//����R+G+B��������������������ratio������ֵ���������ֵThreshold
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
	//����R+G+B������ֵ�����е�ľ�ֵ
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
	//�õ���ֵ
	AvgB /= cnt;
	AvgG /= cnt;
	AvgR /= cnt;
	//����0-255���õ��¾���dst
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
	imgShowSave(dst, "ɫ��ƽ���");
}

void ImgProcess::colorBalance(int deltaR, int deltaG, int deltaB)
{
	Mat cbImg; // ���ڴ���ɫ��ƽ����ͼ��
	cbImg.create(myImg.size(), myImg.type());
	// ��ʼ��ȫ����0
	cbImg = cv::Scalar::all(0);

	for (int i = 0; i < imRows; i++)
	{
		auto* src = (uchar*)(myImg.data + myImg.step * i);
		auto* dst = (uchar*)cbImg.data + cbImg.step * i;
		for (int j = 0; j < imCols; j++)
		{
			// ע�⣬opencv��RGB�Ƿ���
			dst[j * channelNum] = saturate_cast<uchar>(src[j * channelNum] + deltaB);
			dst[j * channelNum + 1] = saturate_cast<uchar>(src[j * channelNum + 1] + deltaG);
			dst[j * channelNum + 2] = saturate_cast<uchar>(src[j * channelNum + 2] + deltaR);
		}
	}
	imgShowSave(cbImg, "�Զ���ɫ��ƽ���");
}
