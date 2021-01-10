import numpy as np
import cv2
import os

image = cv2.imread("baboon.png")

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2



sampled_image = np.zeros(image.shape)
new_image = np.zeros(image.shape)
samples = 100
img_vid_array = []
example_vid_array = []
ex_25 = []
first_10 = []
for i in range(samples+1):
    # Generate matrix with size of the image array, with random values ranging from 0 to 255
    random_train_matrix = np.random.randint(0, 256, size=image.shape)
    # Returns
    floaty_train_images = np.greater(image, random_train_matrix)

    # sample
    sampled_image = np.add(sampled_image,  floaty_train_images)

    # normalize
    normalized_sample = sampled_image/sampled_image.max()

    # image format
    new_image = np.asarray(normalized_sample*255).astype(np.uint8)
    filename = str(i) + 'baboon.png'

    cv2.putText(new_image, str(i),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    img_vid_array.append(new_image)
    if i%10==0:
        frames = 10
        for frame in range(frames):
            example_vid_array.append(new_image)
    if i%25==0:
        frames = 10
        for frame in range(frames):
            ex_25.append(new_image)
    if i<11:
        frames = 10
        for frame in range(frames):
            first_10.append(new_image)

print(np.mean(image), np.mean(new_image))

cv2.putText(image, "original",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

height, width, layers = image.shape
size = (width,height)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_vid_array)):
    out.write(img_vid_array[i])
out.write(image)

frames = 10
for frame in range(frames):
    example_vid_array.append(image)

for i in range(len(example_vid_array)):
    out.write(example_vid_array[i])

for frame in range(frames):
    ex_25.append(image)

for i in range(len(ex_25)):
    out.write(ex_25[i])

frames = 50

for frame in range(frames):
    first_10.append(image)

for i in range(len(first_10)):
    out.write(first_10[i])

out.release()


