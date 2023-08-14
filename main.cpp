#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "AppProp.h"

using namespace cv;

// 全局变量
Mat image;          // 存储原始图像
Mat mask;           // 存储创建的掩码
Rect selection;     // 用于选择区域的矩形

// 鼠标回调函数
void onMouse(int event, int x, int y, int flags, void* userdata) {
    static Point origin;

    if (event == EVENT_LBUTTONDOWN) {
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
    }
    else if (event == EVENT_LBUTTONUP) {
        selection.width = x - origin.x;
        selection.height = y - origin.y;

        // 确保矩形宽和高始终为正
        selection &= Rect(0, 0, image.cols, image.rows);

        mask = Mat::zeros(image.size(), CV_8UC1);
        rectangle(mask, selection, Scalar(255), FILLED);

        imshow("Mask", mask);
    }
    else if (event == EVENT_MOUSEMOVE && flags & EVENT_FLAG_LBUTTON) {
        Mat tempImage = image.clone();
        rectangle(tempImage, origin, Point(x, y), Scalar(0, 255, 0), 2);
        imshow("Original Image", tempImage);
    }
}

int main() {
    // 读取图像
    image = imread("/home/liutao/CLionProjects/AppProp/data/build.png");
    resize(image,image,Size(image.cols/2,image.rows/2));
    if (image.empty()) {
        std::cerr << "Error: Could not read the image." << std::endl;
        return -1;
    }

    namedWindow("Original Image");
    setMouseCallback("Original Image", onMouse);

    imshow("Original Image", image);
    std::cout << "Drag and release the mouse to select a region." << std::endl;

    while (waitKey(30) != 32) {
    }
    AppProp appProp(image,mask,20);

    return 0;
}
