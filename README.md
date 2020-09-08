# Float-to-Tsetlin
Trying to pass floating variables to the Tsetlin Machine

To be tested.

Binarization techniqe that uses the value of each input as a parameter that defines an unfair coinflip as its input, for each epoch.

Example:
For a grayscale image, each pixel gray value works as an unfair coinflip, based in the range of values a pixel can have. 
Psudeocode for how this works for one pixel:
```
def coinflip_pixel(pixel):
  r=random.randint(0,255)
  return r>pixel
``` 
With multiple epochs, the average value for an input will approach it's value, in the the normalized input space. This average can then again be mapped to it's original input space.

Video example of this method with baboon.png :  https://youtu.be/t5Rt4CBOeDc
