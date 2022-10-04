---
layout: about
title: about
permalink: /
subtitle: <a href='http://jtheiner.de'>Jonas Theiner</a> and Ralph Ewerth

profile:
  align: 
  image: teaser_wld.png
  image_circular: false # crops the image to make it circular
  caption-top: >
    <span class="badge font-weight-bold success-color-dark text-uppercase align-middle"> TLTR </span><b>  Learn individual camera parameters from segment correspondences (lines, point clouds) of a known calibration object by iteratively minimizing the segment reprojection error without relying on keypoint correspondences.</b>
  caption-bottom: >
    <p>$$\hat \phi=\{FoV=41.9°, \mathbf{t}=[-0.1\,m, 60.7\,m, -21.2\,m], pan=14.3°, tilt=61.9°, roll=-0.1°\}\quad \hat \psi=\{k_1=0.198, k_2=0.056\}$$</p>

news: true  # includes a list of news items
selected_papers: true # includes a list of papers marked as "selected={true}"
social: false  # includes social icons at the bottom of the page

workflow:
  image: workflow_wide_v2.svg
---

	
Sports field registration in broadcast videos is typically interpreted as the task of homography
estimation, which provides a mapping between a planar field and the corresponding visible area of the
image.
In contrast to previous approaches, we consider the task as a camera calibration problem.

First, we introduce a differentiable objective function that is able to learn the camera pose and focal
length from segment correspondences (e.g., lines, point clouds), based on pixel-level annotations for
segments of a known calibration object, i.e., the sports field.
The calibration module iteratively minimizes the segment reprojection error induced by the estimated
camera parameters $$\phi$$ and potential lens distortion coefficients $$\psi$$.

Second, we propose a novel approach for 3D sports field registration from broadcast soccer images.
Compared to the typical solution, which subsequently refines an initial estimation, our solution does it
in one step.

The proposed method is evaluated for sports field registration on two datasets
and achieves superior results compared to two state-of-the-art approaches.
