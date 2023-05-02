// ########################################################################################################################### 
// CharTrain.cpp : 
// 
// Name:    Thomas Younghoon Kiel
// Date:    3 / 1 / 2023
// 
// Description: This program implements a supervised learning for computer to recognize the numbers (1-9) and the alphabets (A-Z). 
//              The images for the training characters are provided in the traing_chars.png. The program will detect each char in 
//              the image file and present it to the user for classification of the char (labeling). Then the classification data 
//              and the training/referencing char are saved into XML files for matching program on the K Nearest Neighbors logic  
// 
// ########################################################################################################################### 


#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

///////////////////////////////////////////////////////////////////////////////////////////////////
using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    cv::Mat imgTrainingNumbers;         // input image
    cv::Mat imgGrayscale;               // 
    cv::Mat imgBlurred;                 // declare various images
    cv::Mat imgThresh;                  //
    cv::Mat imgThreshCopy;              //

    std::vector<std::vector<cv::Point> > ptContours;        // declare contours vector
    std::vector<cv::Vec4i> v4iHierarchy;                    // declare contours hierarchy

    // these our training classifications
    cv::Mat matClassificationInts;      

    // these are our training images. due to the data types that the KNN object KNearest requires, 
    // we have to declare a single Mat, then append to it as though it's a vector
    cv::Mat matTrainingImagesAsFlattenedFloats; 

    // possible training chars are digits 0 through 9 and capital letters A through Z, put these in vector intValidChars
    std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
        'U', 'V', 'W', 'X', 'Y', 'Z' };

    // source image file to read in training numbers from
    imgTrainingNumbers = cv::imread("training_chars.png");          

    // trap for missing source image file
    if (imgTrainingNumbers.empty()) {                               
        std::cout << "error: image not read from file\n\n";
        return 0;
    }

    // convert the source image to grayscale
    cv::cvtColor(imgTrainingNumbers, imgGrayscale, COLOR_BGR2GRAY);        

    // blur the image
    cv::GaussianBlur(imgGrayscale,              // input image
        imgBlurred,                             // output image
        cv::Size(5, 5),                         // smoothing window width and height in pixels
        0);                                     // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

    // filter the source image from grayscale to black and white
    cv::adaptiveThreshold(imgBlurred,           // input image
        imgThresh,                              // output image
        255,                                    // make pixels that pass the threshold full white
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,         // use gaussian rather than mean, seems to give better results
        cv::THRESH_BINARY_INV,                  // invert so foreground will be white, background will be black
        11,                                     // size of a pixel neighborhood used to calculate threshold value
        2);                                     // constant subtracted from the mean or weighted mean

    // make a copy of the thresh image, this in necessary because findContours modifies the image
    imgThreshCopy = imgThresh.clone();          

    // find all contours in the source image file and store in a vector 
    cv::findContours(imgThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
        ptContours,                             // output contours
        v4iHierarchy,                           // output hierarchy
        cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
        cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points

    // go through all contours in the vector
    for (int i = 0; i < ptContours.size(); i++) {

        // process only the contour is big enough to consider, ignore noises
        if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {

            // get the bounding rect of the contour
            cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                

            // draw red rectangle around each contour as we ask user for input
            cv::rectangle(imgTrainingNumbers, boundingRect, cv::Scalar(0, 0, 255), 2);      

            // get region of interest (ROI) image of bounding rect
            cv::Mat matROI = imgThresh(boundingRect);           

            // resize image to make it more consistent for recognition and storage
            cv::Mat matROIResized;
            cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     

            // display the images for user's input 
            cv::imshow("matROI", matROI);                               // show ROI image for reference
            cv::imshow("matROIResized", matROIResized);                 // show resized ROI image for reference
            cv::imshow("imgTrainingNumbers", imgTrainingNumbers);       // show training numbers image, this will now have red rectangles drawn on it

            int intChar = cv::waitKey(0);           // get key press

            // if esc key was pressed, then exit program
            if (intChar == 27) {     
                return 0;
            }
            // else if the char entered by the user is in the list of chars we are looking for . . .
            else if (std::find(intValidChars.begin(), intValidChars.end(), intChar) != intValidChars.end()) {     

                // append classification char to integer list of chars
                matClassificationInts.push_back(intChar);       

                // now add the training image after converting Mat to float due to KNearest data types being float
                cv::Mat matImageFloat;                          
                matROIResized.convertTo(matImageFloat, CV_32FC1);       

                // flatten the source array (20x30) to 1x1 row
                cv::Mat matImageFlattenedFloat = matImageFloat.reshape(1, 1);       

                // add to Mat as though it was a vector, this is necessary due to the
                // data types that KNearest.train accepts
                matTrainingImagesAsFlattenedFloats.push_back(matImageFlattenedFloat);       
                                                                                            
            }   
        }   
    }   

    std::cout << "training complete\n\n";
     

    //*********************************************************************************************************
    // save the classifications to a file
    // 
    // open the classifications file
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::WRITE);           

    // trap for error
    if (fsClassifications.isOpened() == false) {
        std::cout << "error, unable to open training classifications file, exiting program\n\n";
        return 0;
    }

    // write classifications into classifications section of classifications file
    fsClassifications << "classifications" << matClassificationInts;

    // close the classifications file
    fsClassifications.release();                                            

    //*********************************************************************************************************
    // save training images to a file 
    //
    // open the training images file
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::WRITE);         

    // trap for error
    if (fsTrainingImages.isOpened() == false) {
        std::cout << "error, unable to open training images file, exiting program\n\n";
        return 0;
    }

    // write training images into images section of images file
    fsTrainingImages << "images" << matTrainingImagesAsFlattenedFloats;

    // close the training images file
    fsTrainingImages.release();                                                 

    return 0;
}



