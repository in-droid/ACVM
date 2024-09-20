# ACVM


## Optical Flow

Optical flow refers to the pattern of apparent motion of objects, surfaces, or edges in a visual scene, caused by the relative motion between the observer (e.g., a camera) and the scene. It is a fundamental concept in computer vision, often used in motion detection, video tracking, and image registration.

### Lucas-Kanade Method[^1]
The **Lucas-Kanade** method is a differential technique for optical flow estimation. It assumes that the flow is essentially constant within a small neighborhood of each pixel. By solving a set of linear equations for each pixel in the neighborhood, the method estimates the motion between two consecutive frames. This approach is efficient and works well for small motions and smooth flow fields.

### Horn-Schunck Method[^2]

The **Horn-Schunck** method is another differential technique for estimating optical flow. Unlike Lucas-Kanade, it imposes a global smoothness constraint on the flow field, ensuring that the estimated flow varies smoothly across the image. This method solves an optimization problem that balances the data fidelity term with the smoothness constraint, making it more suitable for scenarios with larger displacements or less textured regions.

Both methods have their strengths and are widely used in various applications of optical flow estimation.

### Example of Horn-Schunck method
https://github.com/user-attachments/assets/d0869c52-de12-47cb-a53b-7642cb9ba35f


## Mean-shift tracking[^3]
The Mean Shift algorithm works by iteratively shifting a window (or kernel) towards the region of highest density in the feature space, often represented by color histograms. This process is repeated until convergence, meaning the window stabilizes over the target object.

https://github.com/user-attachments/assets/0dbb08b5-58a8-4493-a3ed-a455ff6cf7fe


## Discriminative tracking: MOSSE [^4]
The MOSSE algorithm uses adaptive correlation filters to track an object across frames in a video sequence. It works by learning a filter from a single frame and then applying this filter to subsequent frames to locate the object. The filter is updated continuously to adapt to changes in the object's appearance due to lighting, scale, or rotation.

https://github.com/user-attachments/assets/602d3d28-d338-42d6-93c1-ba51b9d0eb07

## Recursive Bayesian filters: Particle filter[^5]
The Particle Tracker operates by representing the probability distribution of the target's state (e.g., position, velocity) using a set of particles. Each particle represents a possible state of the target and is weighted based on how well it matches the observed data. The algorithm recursively updates the particles over time, using a combination of prediction (based on a motion model) and correction (based on new observations). This allows the tracker to estimate the most likely state of the target in each frame.

https://github.com/user-attachments/assets/5259fb0e-d199-404b-9e0d-0fb228c420b4


## Long-term tracking
Long-term tracking involves monitoring objects over extended timeframes, even when they are briefly out of the camera's sight. A widely used method for long-term tracking is SiamFC (Siam Fully Convolutional), which utilizes a siamese network to assess the similarity between the target object and potential regions in later frames. To manage the reappearance of objects, re-detection techniques are implemented, allowing the tracker to maintain precise tracking over long periods.


https://github.com/user-attachments/assets/d5f5208f-55b1-4e28-a37f-a1e022fd15b4




## References
[^1]: [Lucas-Kanade Method](https://www.researchgate.net/publication/215458777_An_Iterative_Image_Registration_Technique_with_an_Application_to_Stereo_Vision_IJCAI)
[^2]: [Horn-Schunck Method](https://www.sciencedirect.com/science/article/abs/pii/0004370281900242)
[^3]: [Mean-Shift tracking](https://comaniciu.net/Papers/KernelTracking.pdf)
[^4]: [MOSSEtracker](https://www.researchgate.net/publication/221362729_Visual_object_tracking_using_adaptive_correlation_filters)
[^5]: [Particle filter tracker](https://vision.ee.ethz.ch/publications/get_abstract.cgi?articles=247&mode=&lang=en)
















