# Sampling-Input-Tsetlin-Machine

A Binarization technique that uses the value of each input as a parameter that defines an unfair coin flip as its input, for each epoch.

Example:
For a grayscale image, each pixel gray value works as an unfair coin flip, based in the range of values a pixel can have.
Pseudocode for how this works for one pixel:
```
def coinflip_pixel(pixel):
  r=random.randint(0,255)
  return r>pixel
```
With multiple epochs, the average value for an input will approach it's value, in the the normalized input space. This average can then again be mapped to it's original input space.

Video example of this method with baboon.png :  https://youtu.be/t5Rt4CBOeDc


To be tested.

Based on the assumption that the tsetlin machine can average an input.

Best result so far:
87.8 % average accuracy over the last 100 epochs, on CIFAR10 color images of the automobile - cat pair. It used CTM with clauses = 4000, T = 75, s = 10.0, mask = (32, 32)
