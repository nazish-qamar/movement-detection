package com.example.integrateopencv;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import android.content.pm.PackageManager;
import android.os.Bundle;
import android.Manifest;

import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainActivity extends CameraActivity{

    CameraBridgeViewBase cameraBridgeViewBase;
    //cascade classifier for detecting faces
    Mat curr_gray, prev_gray, rgb, diff;
    List<MatOfPoint> cnts; //for storing contours
    Boolean is_init;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getPermission();
        is_init = false;
        cameraBridgeViewBase = findViewById(R.id.cameraView);
        //set lister
        cameraBridgeViewBase.setCvCameraViewListener(new CameraBridgeViewBase.CvCameraViewListener2(){
            @Override
            public void onCameraViewStarted(int width, int height) {
                curr_gray = new Mat();
                prev_gray = new Mat();
                rgb = new Mat();
                diff = new Mat();
                cnts = new ArrayList<>();
            }

            @Override
            public void onCameraViewStopped() {
                rgb.release();
                curr_gray.release();
                prev_gray.release();
            }

            //this is called when the frame is captured by device's camera
            @Override
            public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
                if(!is_init){ // to check if initial frame is initialized
                    prev_gray = inputFrame.gray();
                    is_init = true;
                    return prev_gray;
                }

                // read rgb
                rgb = inputFrame.rgba();
                curr_gray = inputFrame.gray();
                //detect noise in the frame
                Core.absdiff(curr_gray, prev_gray, diff);
                //convert to 0 or 1 value based on threshold 40
                Imgproc.threshold(diff, diff, 40, 255, Imgproc.THRESH_BINARY);

                //detect contours around the boundary
                // mode = Imgproc.RETR_TREE
                // method = CHAIN_APPROX_SIMPLE - to return only the vertices points to same memory
                Imgproc.findContours(diff, cnts, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

                //drawing contours
                // contourIdx = -1, to draw all contours
                Imgproc.drawContours(rgb, cnts, -1, new Scalar(255, 0, 0), 4);

                // draw bounding rectangles around contours
                for(MatOfPoint m: cnts){
                    Rect r = Imgproc.boundingRect(m);
                    Imgproc.rectangle(rgb, r, new Scalar(0, 0, 255), 3);
                }
                cnts.clear(); // need to clear otherwise it will keep on adding new contours on top of older ones
                prev_gray = curr_gray.clone();
                return rgb;
            }
        });

        if(OpenCVLoader.initDebug()) { //to check openCV is successfully loaded
            cameraBridgeViewBase.enableView();

        }
    }

    //resume when the app is in the foreground
    @Override
    protected void onResume() {
        super.onResume();
        cameraBridgeViewBase.enableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraBridgeViewBase.disableView();
    }

    //pause view when the app is in the background

    @Override
    protected void onPause() {
        super.onPause();
        cameraBridgeViewBase.disableView();
    }


    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(cameraBridgeViewBase);
    }

    private void getPermission() {
        if(checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 101);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(grantResults.length>0 && grantResults[0]!=PackageManager.PERMISSION_GRANTED){
            getPermission();
        }
    }
}