#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace cv;
using namespace cv::ml;
using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


bool isCircle(const Mat& roiImage, const Mat& temp) {

    GaussianBlur(temp, temp, Size(5, 5), 0, 0);
    vector<Vec3f> circles;
    HoughCircles(temp, circles, HOUGH_GRADIENT, 1, temp.rows / 10, 80, 20, 5, 200);
    return !circles.empty();
}


void trafficSignsDetection(Mat src, int op) {

    Mat copy;
    src.copyTo(copy);

    int width = src.cols;
    int height = src.rows;

    Mat Mat_hsv;
    cvtColor(copy, Mat_hsv, COLOR_BGR2HSV);


    Mat Mat_rgb_red = Mat::zeros(src.size(), CV_8UC1);
    Mat Mat_rgb_blue = Mat::zeros(src.size(), CV_8UC1);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {

            Vec3b hsv = Mat_hsv.at<Vec3b>(i, j);
            float H = hsv[0];
            float S = hsv[1];
            float V = hsv[2];

            if (((H >= 0 && H <= 15) || (H >= 165 && H <= 180)) && S >= 50 && S <= 255 && V >= 50 && V <= 255)
            {
                Mat_rgb_red.at<uchar>(i, j) = 255;
            }

            if (H >= 90 && H <= 130 && S >= 50 && S <= 255 && V >= 50 && V <= 255)

            {  
                Mat_rgb_blue.at<uchar>(i, j) = 255;
            }

        }
    }
 
    medianBlur(Mat_rgb_red, Mat_rgb_red, 3);
    medianBlur(Mat_rgb_blue, Mat_rgb_blue, 3);

    Mat elipsa1 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
    Mat elipsa2 = getStructuringElement(MORPH_ELLIPSE, Size(2 * 2 + 1, 2 * 2 + 1), Point(2, 2));

    erode(Mat_rgb_red, Mat_rgb_red, elipsa1);
    erode(Mat_rgb_blue, Mat_rgb_blue, elipsa1);

    dilate(Mat_rgb_red, Mat_rgb_red, elipsa2);
    dilate(Mat_rgb_blue, Mat_rgb_blue, elipsa2);
 
    vector<vector<Point>> contoursRed;
    vector<vector<Point>> contoursBlue;

    vector<Vec4i> hierarchyRed;
    vector<Vec4i> hierarchyBlue;

    findContours(Mat_rgb_red, contoursRed, hierarchyRed, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    findContours(Mat_rgb_blue, contoursBlue, hierarchyBlue, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

    vector<vector<Point>> contours_polyRed(contoursRed.size());
    vector<Rect> boundRectRed(contoursRed.size());
    vector<Point2f> centruRed(contoursRed.size());
    vector<float> razaRed(contoursRed.size());

    vector<vector<Point>> contours_polyBlue(contoursBlue.size());
    vector<Rect> boundRectBlue(contoursBlue.size());
    vector<Point2f> centruBlue(contoursBlue.size());
    vector<float> razaBlue(contoursBlue.size());

    for (size_t i = 0; i < contoursRed.size(); i++) {
        approxPolyDP(Mat(contoursRed[i]), contours_polyRed[i], 2, true);
        boundRectRed[i] = boundingRect(Mat(contours_polyRed[i]));
        minEnclosingCircle(contours_polyRed[i], centruRed[i], razaRed[i]);
    }

    for (size_t i = 0; i < contoursBlue.size(); i++) {
        approxPolyDP(Mat(contoursBlue[i]), contours_polyBlue[i], 2, true);
        boundRectBlue[i] = boundingRect(Mat(contours_polyBlue[i]));
        minEnclosingCircle(contours_polyBlue[i], centruBlue[i], razaBlue[i]);
    
    }


    if (op == 1) 
    {
        cout << "For RED detection: " << endl;
        cout << contours_polyRed[0] << centruRed[0] << razaRed[0] << endl;

        cout << "For BLUE detection: " << endl;
        cout << contours_polyBlue[0] << centruBlue[0] << razaBlue[0] << endl;
    }


    for (size_t i = 0; i < contoursRed.size(); i++)
    {
        Rect rect1 = boundRectRed[i];
        float ratio1 = static_cast<float>(rect1.width) / static_cast<float>(rect1.height);
        float Area1 = static_cast<float>(rect1.width) * static_cast<float>(rect1.height);
        float arie1 = static_cast<float>(contourArea(contoursRed[i]));
        float perimetru1 = static_cast<float>(arcLength(contoursRed[i], true));


        if (arie1 < 400 || ratio1 > 1.5 || ratio1 < 0.8)
            continue;


        Mat roiImage = Mat_rgb_red(rect1).clone();
        Mat temp = roiImage.clone();

        bool iscircle = isCircle(roiImage, temp);

        if(op==1) cout << "circle:" << iscircle << endl;

        if (!iscircle)
            continue;

        float C1 = (4 * M_PI * arie1) / (perimetru1 * perimetru1);

        if (C1 < 0.4)
            continue;

        Scalar color(255, 0, 255);
        drawContours(src, contours_polyRed, i, color, 2, LINE_8, hierarchyRed, 0, Point());

    }

    for (size_t i = 0; i < contoursBlue.size(); i++)
    {
        Rect rect2 = boundRectBlue[i];
        float ratio2 = static_cast<float>(rect2.width) / static_cast<float>(rect2.height);
        float Area2 = static_cast<float>(rect2.width) * static_cast<float>(rect2.height);
        float arie2 = static_cast<float>(contourArea(contoursBlue[i]));
        float perimetru2 = static_cast<float>(arcLength(contoursBlue[i], true));


        if (arie2 < 400 || ratio2 > 1.5 || ratio2 < 0.8)
            continue;


        Mat roiImage = Mat_rgb_blue(rect2).clone();
        Mat temp = roiImage.clone();

        bool iscircle = isCircle(roiImage, temp);
        if(op==1) cout << "circle:" << iscircle << endl;

        if (!iscircle)
            continue;


        float C2 = (4 * M_PI * arie2) / (perimetru2 * perimetru2);

        if (C2 < 0.4)
            continue;

        Scalar color(255, 0, 255);
        drawContours(src, contours_polyBlue, i, color, 2, LINE_8, hierarchyBlue, 0, Point());

    }
}


int main() {

    int op;
    do {

        printf("Alegeti optiunea dorita:\n");
        printf(" 1 - Detectati semnele de circulatie folosing imagini de test\n");
        printf(" 2 - Detectati semnele folosind un video\n");

        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);

        if (op == 1) 
        {
            char path[50];

            for (int k = 0; k <= 7; k++) {
                sprintf(path, "raw/%d.jpg", k + 1);
                cout << path << endl;

                Mat src = imread(path);
                
                cv::imshow("Source", src);
                waitKey(0);

                if (src.empty()) {
                    cerr << "Failed to load image: " << path << endl;
                    return 1;
                }

                if (src.channels() == 1) {
                    cvtColor(src, src, COLOR_GRAY2BGR);
                }
                trafficSignsDetection(src, op);

                cv::imshow("Destination", src);
                waitKey(0);
            }
        }

        else
            if (op == 2)
            { 
                VideoCapture cap(0);

                if (!cap.isOpened())
                {
                    std::cout << "Failed to open camera!" << std::endl;
                    return -1;
                }

                while (true)
                {
                    Mat frame;
                    cap >> frame;

                    if (frame.empty())
                    {
                        std::cout << "Failed to capture frame!" << std::endl;
                        break;
                    }

                    trafficSignsDetection(frame, op);

                    Mat rotatedFrame;
                    cv::rotate(frame, rotatedFrame, ROTATE_180);
                    cv::imshow("Camera", rotatedFrame);

                    if (waitKey(1) == 'q')
                    {
                        break;
                    }
                }

                cap.release();
                cv::destroyAllWindows();
            }

            else

                if (op != 0 && op != 1 && op != 2)
                {
                    cout << "Optiune invalida" << endl;
                }

    } while (op != 0);
    return 0;
}