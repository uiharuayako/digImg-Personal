#include "ImgProcess.h"
int main() {
//	ImgProcess test("20180620-tianjin-2076x2816x3BIP", 2076, 2816, 1);
	ImgProcess test("ik_beijing_p.bmp");
	test.imgBinOTSU();

	waitKey();
}