
# importing openCV library
import cv2
import os


# function to read the images by taking there path
def read_image(path1, path2):
    read_img1 = cv2.imread(path1)
    images = []
    for image in os.listdir(path2):
        images.append(cv2.imread(path2 + image))
    return read_img1, images


# function to convert images from RGB to gray scale
def convert_to_grayscale(pic1, pic2):
    gray_img1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
    return gray_img1, gray_img2


# function to detect the features by finding key points
# and descriptors from the image
def detector(image1, images):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    keypoint1, descriptors1 = sift.detectAndCompute(image1, None)
    kps, descs = [], []
    for image in images:
        kp, des = sift.detectAndCompute(image, None)
        kps.append(kp)
        descs.append(des)
    return keypoint1, descriptors1, kps, descs


# function to find best detected features using flann
# matcher and match them according to their humming distance
def flann_matcher(queryDes, trainDes):
    flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict(checks=50))

    # add descriptors
    flann.add(trainDes)

    matches = flann.knnMatch(queryDes, trainDes, k=len(trainDes))

    # finding the humming distance of the matches and sorting them
    good_matches = [[0, 0] for i in range(len(matches))]

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.1*n.distance:
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
    second_image_path = "images/"

    # reading the image from there paths
    img1, imgs = read_image(first_image_path, second_image_path)
    #img2 = cv2.rotate(img2, cv2.ROTATE_180)

    # converting the read images into the gray scale images
    #gray_pic1, gray_pic2 = convert_to_grayscale(img1, img2)

    # storing the found key points and descriptors of both of the images
    key_pt1, descrip1, key_pts, descrips = detector(img1, imgs)

    # sorting the number of best matches obtained from brute force matcher
    mt, gm = flann_matcher(descrip1, descrips)
    tot_feature_matches = gm.count([1, 0])
    print(f'Total Number of Features matches found are {tot_feature_matches}')

    # after drawing the feature matches displaying the output image
    display_output(img1, key_pt1, imgs, key_pts, mt, gm)
    cv2.waitKey()
    cv2.destroyAllWindows()
