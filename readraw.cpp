// DIP.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <iostream>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>   

using namespace std;  //使用cout输出方式
using namespace cv;   // 省去函数前面加cv::的必要性


int main()
{
    FILE* fp;

    if (fopen_s(&fp, "20180620-tianjin-2076x2816x3BIP", "rb") != 0)
    {
        printf("cannot open file for read\n");
        waitKey();//opencv中常用waitKey(n),n<=0表示一直
        exit(0);
    }

    
    int cols = 2076;
    int rows = 2816;
    int bands = 3;// 波段数
    int pixels = cols * rows * bands;
    
    unsigned char* data = new uchar[pixels*sizeof(uchar)];//uchar等效于char，此处可以借鉴这样的预分配内存方法
    fread(data, sizeof(uchar), pixels, fp);//fread函数的用法
    fclose(fp);

    Mat M(rows, cols, CV_8UC3, Scalar(0, 0, 0));
    unsigned char* ptr = M.data;//data是存数据的部分指针

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < 3; k++)
            {
                ptr[(i * cols + j) * 3 + k] = data[(i * cols + j) * bands + (k + 1)];//BIP
                //ptr[(i * cols + j) * 3 + k] = data[i * cols * bands + (k + 3) * cols + j];//BIL
                //ptr[(i * cols + j) * 3 + k] = data[(i + rows * k) * cols + j];//BSQ
            }
        }
    }

    namedWindow("image", 0);//1为按图片大小显示，0为跟据窗口大小调整
    imshow("image", M);  // 显示图片 
    waitKey();
    imwrite("pic.bmp", M); // 存为bmp格式图片
    return 0;

}

