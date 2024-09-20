# ACVM


## Optical Flow

Optical flow refers to the pattern of apparent motion of objects, surfaces, or edges in a visual scene, caused by the relative motion between the observer (e.g., a camera) and the scene. It is a fundamental concept in computer vision, often used in motion detection, video tracking, and image registration.

### Lucas-Kanade Method[^1]
The **Lucas-Kanade** method is a differential technique for optical flow estimation. It assumes that the flow is essentially constant within a small neighborhood of each pixel. By solving a set of linear equations for each pixel in the neighborhood, the method estimates the motion between two consecutive frames. This approach is efficient and works well for small motions and smooth flow fields.

### Horn-Schunck Method[^2]

The **Horn-Schunck** method is another differential technique for estimating optical flow. Unlike Lucas-Kanade, it imposes a global smoothness constraint on the flow field, ensuring that the estimated flow varies smoothly across the image. This method solves an optimization problem that balances the data fidelity term with the smoothness constraint, making it more suitable for scenarios with larger displacements or less textured regions.

Both methods have their strengths and are widely used in various applications of optical flow estimation.

### Example of Horn-Schunck method
https://github.com/user-attachments/assets/05171b02-ef6d-4d84-8cac-17c65fc7f7e6



## Mean-shift tracking[^3]
The Mean Shift algorithm works by iteratively shifting a window (or kernel) towards the region of highest density in the feature space, often represented by color histograms. This process is repeated until convergence, meaning the window stabilizes over the target object.

https://github.com/user-attachments/assets/aee17fc8-23b8-41e1-87d6-7ed134b752af


## Long-term tracking
https://github.com/user-attachments/assets/195179ab-028f-435f-974a-1aad4edf6fc5

##


https://github.com/user-attachments/assets/0dbb08b5-58a8-4493-a3ed-a455ff6cf7fe


## References
[^1]: [Lucas-Kanade Method](https://www.researchgate.net/publication/215458777_An_Iterative_Image_Registration_Technique_with_an_Application_to_Stereo_Vision_IJCAI)
[^2]: [Horn-Schunck Method](https://www.sciencedirect.com/science/article/abs/pii/0004370281900242)
[^3]: [Mean-Shift tracking](https://comaniciu.net/Papers/KernelTracking.pdf)












