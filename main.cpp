#include "ImgProcess.h"
int main() {
	// �ṩӦ�ó������ں�һ�������н���
	// ͼԴ��������ͼ��
	// ��һ�ţ���ͼ����ȡrawͼ�������
	ImgProcess hugeImg("20180620-tianjin-2076x2816x3BIP", 2076, 2816, 1);
	// �ڶ��ţ��е�ͼ������ʾ��ͼ��
	ImgProcess midImg("ik_beijing_p.bmp");
	// ���һ�ţ���Сͼ�񣬽�������ʾ����غ���������
	ImgProcess minImg("ik_beijing_c_min.jpg");
	int cho = -1;
	while (cho != 0) {
		cout << "===================" << endl;
		cout << "======��ӭʹ��!======" << endl;
		cout << "=======Menu========" << endl;
		cout << "1.������1��ͼ������㣺�Ҷ����Ա任" << endl;
		cout << "2.������2��ͼ��ֲ�������ͨ�˲�����ͨ�˲�����ֵ�˲�" << endl;
		cout << "3.��ѡ��1��ͼ��ļ��δ���ƽ�ơ����š���ת" << endl;
		cout << "4.��ѡ��2��ͼ���ֵ����״̬�����жϷ�����" << endl;
		cout << "5.��ѡ��3������ͼ�������غ���������" << endl;
		cout << "6.��ѡ��4��α��ɫ��ǿ" << endl;
		cout << "7.��ѡ��5��ɫ��ƽ��" << endl;
		cout << "0. �˳�����" << endl;
		cout << "���ѡ���ǣ�";
		cin >> cho;
		// �����Ļ
		system("cls");
		// ����һЩ��ʱ����
		double k, b;
		int cho2, dr, dg, db;
		switch (cho) {
		case 0:
			exit(0);
		case 1:
			cout << "�ر�ͼ�񴰿���������һ�����" << endl;
			cout << "======��ӭʹ��!======" << endl;
			cout << "=====�Ҷ����Ա任=====" << endl;
			cout << "�������Ա任������б�ʣ�k����";
			k = 0;
			cin >> k;
			cout << endl << "�������Ա任�����Ľؾࣨb����";
			b = 0;
			cin >> b;
			hugeImg.grayLinear(k, b);
			cout << endl << "������ϣ�" << endl;
			break;
		case 2:
			cout << "�ر�ͼ�񴰿���������һ�����" << endl;
			cout << "======��ӭʹ��!======" << endl;
			cout << "=====ͼ��ֲ�����=====" << endl;
			cout << "ѡ���ͨ�˲��ķ�ʽ��1.�鱾H1���� 2.�鱾H2��������������˹�� 3.���������˹����";
			cin >> cho2;
			hugeImg.highPass(cho2);
			hugeImg.lowPass();
			hugeImg.medPass();
			cout << endl << "������ϣ�" << endl;
			break;
		case 3:
			cout << "�ر�ͼ�񴰿���������һ�����" << endl;
			cout << "======��ӭʹ��!======" << endl;
			cout << "=====ͼ�񼸺δ���=====" << endl;
			cout << "ѡ�񼸺δ���ķ�ʽ(1.ƽ�� 2.���� 3.��ת)��";
			cin >> cho2;
			switch (cho2)
			{
			case 1:
				cout << endl << "�����Σ���˳������x��y����ƫ�������ֵ" << endl;
				int dx, dy;
				cin >> dx;
				cin >> dy;
				hugeImg.imgMove(dx, dy, 0);
				hugeImg.imgMove(dx, dy, 1);
				break;
			case 2:
				cout << endl << "�����Σ���˳������x��y��������ű���" << endl;
				int rx, ry;
				cin >> rx;
				cin >> ry;
				hugeImg.imgZoom(rx, ry);
				break;
				cout << endl << "����Ƕ�ֵ����ת�Ƕȣ�˳ʱ����ת��" << endl;
				double ang;
				hugeImg.imgRotate(ang);
				break;
			}
			cout << endl << "������ϣ�" << endl;
			break;
		case 4:
			cout << "�ر�ͼ�񴰿���������һ�����" << endl;
			cout << "======��ӭʹ��!======" << endl;
			cout << "=====ͼ��Ķ�ֵ��=====" << endl;
			midImg.imgBinOTSU();
			midImg.imgBinRatio();
			break;
		case 5:
			cout << "�ر�ͼ�񴰿���������һ�����" << endl;
			cout << "======��ӭʹ��!======" << endl;
			cout << "====ͼ������ط���====" << endl;
			minImg.imgAutoCorr();
			cout << endl << "������ϣ�" << endl;
			break;
		case 6:
			cout << "�ر�ͼ�񴰿���������һ�����" << endl;
			cout << "======��ӭʹ��!======" << endl;
			cout << "====ͼ��α��ɫ��ǿ====" << endl;
			hugeImg.bin2Color();
			cout << endl << "������ϣ�" << endl;
			break;
		case 7:
			cout << "�ر�ͼ�񴰿���������һ�����" << endl;
			cout << "======��ӭʹ��!======" << endl;
			cout << "====ͼ���ɫ��ƽ��====" << endl;
			cout << "ѡ��ɫ��ƽ��ķ�ʽ(1.�Զ��� 2.��ƽ��)��";
			cin >> cho2;
			switch (cho2)
			{
			case 1:
				cout << endl << "�����Σ���˳������RGB�ı��ֵ" << endl;
				cin >> dr;
				cin >> dg;
				cin >> db;
				hugeImg.colorBalance(dr, dg, db);
				break;
			case 2:
				hugeImg.whiteBalance();
				break;
			}
			cout << endl << "������ϣ�" << endl;
			break;
		default:
			cout << "����Խ�磡����������" << endl;
			system("pause");
		}
		cout << "�ر�����ͼ�񴰿��Բ���������Ŀ" << endl;
		system("pause");
		system("cls");
	}
	waitKey();
}