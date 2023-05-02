// ########################################################################################################################### 
// CharMatch.cpp : 
// 
// Name:    Thomas Younghoon Kiel
// Date:    3 / 1 / 2023
// 
// Description: This program reads pre-trained classification and image data from XML files. The XML files contains the data for
//              the numbers (1-9) and alphabets (A-Z) in the format of K Nearest Neighbors logic. Then the program captures an input
//              char from a webcam and matches against the pre-trained image data. If the char is found in the image data, 
//              then it outputs the corresponding classification char (label).
// 
// ########################################################################################################################### 

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>

// global variables ///////////////////////////////////////////////////////////////////////////////
const int MIN_CONTOUR_AREA = 100;
const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;


int main() {

    // read in previously trained classifications
    // read the classification numbers into this variable as if it is a vector
    cv::Mat matClassificationInts;      

    // open the classifications file
    cv::FileStorage fsClassifications("classifications.xml", cv::FileStorage::READ);        

    // trap for file errors
    if (fsClassifications.isOpened() == false) {
        std::cout << "error, unable to open training classifications file, exiting program\n\n";
        return(0);
    }

    // read classifications section into Mat classifications variable
    fsClassifications["classifications"] >> matClassificationInts;

    // close the classifications file
    fsClassifications.release();                                        

    // read in training images
    // read multiple images into this single image variable as if it is a vector
    cv::Mat matTrainingImagesAsFlattenedFloats;

    // open the training images file
    cv::FileStorage fsTrainingImages("images.xml", cv::FileStorage::READ);          

    // trap for file error
    if (fsTrainingImages.isOpened() == false) {
        std::cout << "error, unable to open training images file, exiting program\n\n";
        return(0);
    }

    // read images section into Mat training images variable
    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;

    // close the traning images file
    fsTrainingImages.release();                                                 

    // instantiate the KNN object
    cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            

    // load the pre-trained data to train computer
    // note that both parameters have to be of type Mat (a single Mat)
    //      even though in reality they are multiple images and numbers
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

    // match the input from webcam 

    // Open the default camera
    cv::VideoCapture cap(0);

    // trap for error
    if (!cap.isOpened())
    {
        std::cerr << "Unable to open the camera\n";
        return -1;
    }


    cv::Mat frame;                  // read in the test numbers image
    cv::Mat matGrayscale;           // converted test image to grayscale
    cv::Mat matBlurred;             // bluured test image
    cv::Mat matThresh;              // threshed test image
    cv::Mat matThreshCopy;          // copy of the thresh image

    std::string strFinalString;     // declare final string, this will have the final number sequence by the end of the program

    while (true)
    {
        // capture the test char
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty())
        {
            std::cerr << "Unable to capture frame\n";
            break;
        }

        // show the test input image with green boxes drawn around found digits
        cv::imshow("frame", frame);     

        // Check if the user pressed the 'C' or 'c' key for capture and ready for matching
        int key = cv::waitKey(1);

        // no key pressed, then continue
        if (key == 0) {
            continue;
        }

        // esc pressed, then exit program
        if (key == 27)
        {
            break;
        }

        // Check if the user pressed the 'C' or 'c' key for capture ==> user is ready with the test image
        if ((key == 67) || (key == 99))
        {
            // convert to grayscale
            cv::cvtColor(frame, matGrayscale, cv::COLOR_BGR2GRAY);         

            // blur
            cv::GaussianBlur(matGrayscale, // input image
                matBlurred,                // output image
                cv::Size(5, 5),            // smoothing window width and height in pixels
                0);                        // sigma value, determines how much the image will be blurred, zero makes function choose the sigma value

            // filter image from grayscale to black and white
            cv::adaptiveThreshold(matBlurred,         // input image
                matThresh,                            // output image
                255,                                  // make pixels that pass the threshold full white
                cv::ADAPTIVE_THRESH_GAUSSIAN_C,       // use gaussian rather than mean, seems to give better results
                cv::THRESH_BINARY_INV,                // invert so foreground will be white, background will be black
                11,                                   // size of a pixel neighborhood used to calculate threshold value
                2);                                   // constant subtracted from the mean or weighted mean

            // make a copy of the thresh image, this in necessary because findContours modifies the image
            matThreshCopy = matThresh.clone();              

            std::vector<std::vector<cv::Point> > ptContours;        // declare a vector for the contours
            std::vector<cv::Vec4i> v4iHierarchy;                    // declare a vector for the hierarchy (we won't use this in this program but this may be helpful for reference)

            
            cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
                ptContours,                             // output contours
                v4iHierarchy,                           // output hierarchy
                cv::RETR_EXTERNAL,                      // retrieve the outermost contours only
                cv::CHAIN_APPROX_SIMPLE);               // compress horizontal, vertical, and diagonal segments and leave only their end points
            
            
            /*
            // this is to draw a line around the edge
            cv::findContours(matThreshCopy,             // input image, make sure to use a copy since the function will modify this image in the course of finding contours
                ptContours,                             // output contours
                v4iHierarchy,                           // output hierarchy
                cv::RETR_TREE,                          // retrieve all possible contours from the image, compared to outermost contours only
                cv::CHAIN_APPROX_NONE);                 // store all contour points. slower than CHAON_APPROX_SIMPLE
            */

            /*
            // display edges around the input char
             
            cv::Mat image_copy = frame.clone();
            cv::drawContours(image_copy, ptContours, -1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("edges around", image_copy);
            */

            // go through all contours
            for (int i = 0; i < ptContours.size(); i++) {

                // debugging for contours
                // std::cout << "\n\n" << "contour at " << i << " = " << ptContours[i] << "\n\n"; 

                // proceed if contour is big enough to consider
                if (cv::contourArea(ptContours[i]) > MIN_CONTOUR_AREA) {

                    // get the bounding rect
                    cv::Rect boundingRect = cv::boundingRect(ptContours[i]);                
                    
                    // draw red rectangle around each contour as we ask user for input
                    cv::rectangle(frame, boundingRect, cv::Scalar(0, 0, 255), 2);      

                    // get ROI image of bounding rect
                    cv::Mat matROI = matThresh(boundingRect);          
                    cv::imshow("ROI", matROI);     

                    // resize image, this will be more consistent for recognition and storage
                    cv::Mat matROIResized;
                    cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     

                    // convert the input Mat to float, necessary for call to find_nearest
                    cv::Mat matROIFloat;
                    matROIResized.convertTo(matROIFloat, CV_32FC1);             

                    // flatten the source array (20x30) to 1x1 row
                    cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

                    // char the input test char represents
                    cv::Mat matCurrentChar(0, 0, CV_32F);

                    // call find_nearest to find the nearrest matching char in the trained data
                    kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     

                    // find the float data in the Classification file
                    float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

                    // find the ASCII char to display
                    strFinalString = char(int(fltCurrentChar));        

                    // show the ASCII char
                    std::cout << "\n\n" << "numbers read = " << strFinalString << "\n\n";       

                }
            }
        }

        // reset the key variable 
        key = 0;
    }

    cv::destroyAllWindows();

    return 0;
}


