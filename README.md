# Beyond Appearance: Camouflaged Object Detection via Geometric Structure

> [**Beyond Appearance: Camouflaged Object Detection via Geometric Structure**](https://cvpr.thecvf.com/virtual/2026/poster/36862)
>
> **CVPR 2026**
>
> **MDE & COD**

**Abstract**

Depth priors provide salient geometric structure that benefits camouflaged object detection (COD), but directly using Monocular Depth Estimation (MDE) causes a task misalignment that still fails to identify camouflaged objects.
To address this issue, we propose the Depth Segment Anything Model (DepthSAM), a MDE-adapted method specifically designed to mitigate this misalignment.
DepthSAM incorporates two core innovations: (1) a Sparse Mixture-of-Experts Adapter (SMEA) that enables MDE to learn semantic information unique to camouflaged scenes, and (2) a Geometric–Semantic Fusion Module (GSFM) that efficiently integrates geometric cues with high-level semantics. With these components, DepthSAM achieves both robust semantic understanding in camouflaged environments and accurate segmentation of camouflaged objects.
Extensive experiments show that DepthSAM achieves new SOTA performance on three major benchmarks. For example, on COD10K, its $S_{\alpha}$ and $F_{\beta}^{\omega}$ metrics surpass the best competing methods by 3.0\% and 4.3\%, respectively.
<p float="left">
  <img src="Images/Figure1.png?raw=true" width="50%" />
</p>

<p float="left">
  <img src="Images/Figure2.png?raw=true" width="100%" />
</p>
