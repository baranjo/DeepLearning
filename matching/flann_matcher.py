
# importing openCV library
import cv2


# function to read the images by taking there path
def read_image(path1, path2):
    read_img1 = cv2.imread(path1)
    read_img2 = cv2.imread(path2)
    return read_img1, read_img2


# function to convert images from RGB to gray scale
def convert_to_grayscale(pic1, pic2):
    gray_img1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    return gray_img1, gray_img2


# function to detect the features by finding key points
# and descriptors from the image
def detector(image1, image2):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    keypoint1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoint2, descriptors2 = sift.detectAndCompute(image2, None)
    return keypoint1, descriptors1, keypoint2, descriptors2


# function to find best detected features using flann
# matcher and match them according to their humming distance
def flann_matcher(des1, des2):
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=20), dict(checks=150))
    matches = flann.knnMatch(des1, des2, k=2)

    # finding the humming distance of the matches and sorting them
    good_matches = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            good_matches[i] = [1, 0]
    return matches, good_matches


# function displaying the output image with the feature matching
def display_output(pic1, kpt1, pic2, kpt2, matches, good_matches):
    # drawing the feature matches using drawMatches() function
    matched = cv2.drawMatchesKnn(pic1, kpt1, pic2, kpt2, matches[:30], outImg=None, matchesMask=good_matches[:30], flags=2)
    cv2.imshow('Output image', matched)



if __name__ == '__main__':
    # giving the path of both of the images
    first_image_path = "images/image_05644.jpg"
    second_image_path = "images/image_05643.jpg"

    # reading the image from there paths
    img1, img2 = read_image(first_image_path, second_image_path)
    img2 = cv2.rotate(img2, cv2.ROTATE_180)

    # converting the read images into the gray scale images
    gray_pic1, gray_pic2 = convert_to_grayscale(img1, img2)

    # storing the found key points and descriptors of both of the images
    key_pt1, descrip1, key_pt2, descrip2 = detector(gray_pic1, gray_pic2)

    # sorting the number of best matches obtained from brute force matcher
    m, gm = flann_matcher(descrip1, descrip2)
    tot_feature_matches = gm.count([1, 0])
    print(f'Total Number of Features matches found are {tot_feature_matches}')

    # after drawing the feature matches displaying the output image
    display_output(gray_pic1, key_pt1, gray_pic2, key_pt2, m, gm)
    cv2.waitKey()
    cv2.destroyAllWindows()
