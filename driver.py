import laneDetection
import matplotlib.pyplot as plt
import glob
from moviepy.editor import VideoFileClip


# here we will get all the jpg image from the test image and store it in testImages array
def loadTestImage():
    testImage = [path for path in glob.glob("test_images/*.jpg")]
    testImages = [plt.imread(image_path) for image_path in testImage]
    return testImages


def showAllImage(listOfImage):
    """
    it will show all the images with two columns
    """
    laneDetection.imageShow(listOfImage)

def videoTestLaneDetection():
    counter = 0
    output = "CS156_finalProject.mp4"
    testVideo = VideoFileClip("test_videos/challenge.mp4")
    out_clip = testVideo.fl_image(laneDetection.pipeline)
    out_clip.write_videofile(output, audio=False)
    print(counter)


# here we will show the original list of test image, then lane detected image, then show the lane detection on video
def main():
    # load the test image
    testImages = loadTestImage()
    showAllImage(testImages)
    # now let detect the lane
    finalResultImage = [laneDetection.pipeline(image) for image in testImages]
    showAllImage(finalResultImage)
    videoTestLaneDetection()


if __name__ == "__main__":
    main()




