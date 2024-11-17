---
layout: post
title:  "Turn 3D Gaussian Splat Files into Stunning Assets in Unity 6"
author: Gurwinder
categories: [ Game Development, Unity ]
image: assets/images/GS3.png
featured: false
hidden: false
---

This guide walks you through the process of loading splat files in Unity 6 using the Gaussian Splatting Playground Unity plugin. By the end of this tutorial, you’ll be able to load splat files, visualize them, and experiment with Gaussian Splatting parameters in Unity.

### **Prerequisites**

1. **Unity 6 Installed**: Ensure Unity 6 is installed on your system.

2. **Gaussian Splatting Playground Plugin**: Download and integrate the plugin into your Unity project. You can usually find this plugin in the Unity Asset Store or from [GitHub](https://github.com/aras-p/UnityGaussianSplatting).

3. **A Splat File**: Have a `.splat` file ready for visualization. This file contains point cloud or splat data. The original paper [GitHub](https://github.com/graphdeco-inria/gaussian-splatting) has a [Models](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) to 14GB zip of their models.

### **Step 2: Converting a Splat File to Unity Asset**

![walking]({{ site.baseurl }}/assets/images/GS4.png){:style="display:block; margin-left:auto; margin-right:auto"}

To use a splat file effectively in Unity, it’s best to convert it into a Unity-compatible asset.

1. **Open the Gaussian Splat Asset Creation Tool**:
   - Go to **Tools > Gaussian Splats > Create GaussianSplatAsset** in the Unity menu.

2. **Configure the Asset Conversion**:
   - **Input PLY File**: Point this field to your Gaussian Splat `.ply` file. 
     - Note: Ensure the file is a Gaussian Splat `.ply` file. For official paper models, these files are typically located under `point_cloud/iteration_*/point_cloud.ply`.
   - **Optional Camera Configuration**: If you have a `cameras.json` file, place it next to the `.ply` file or in a parent directory. This can enhance how the splats are rendered.

3. **Select Compression Options**:
   - Choose the desired **compression level** based on your needs:
     - **Very Low**: Compresses assets significantly while retaining decent quality.
   - Example: A model compressed at "Very Low" results in an asset size of under 8MB, suitable for many applications.

4. **Choose Output Folder**:
   - Specify where the converted asset will be saved in your Unity project.

5. **Create the Asset**:
   - Press the **Create Asset** button. Unity will process the `.ply` file and generate a GaussianSplat asset ready for use.

---

### **Step 3: Loading and Rendering the Asset**

![walking]({{ site.baseurl }}/assets/images/GS5.png){:style="display:block; margin-left:auto; margin-right:auto"}

1. **Add a Gaussian Splat Renderer**:
   - Create an empty GameObject in the **Hierarchy** and name it `GaussianSplatRenderer`.
   - Attach the `GaussianSplatRenderer` component from the plugin.

2. **Assign the Asset**:
   - Drag and drop the newly created GaussianSplat asset into the `GaussianSplatRenderer`'s **Splat Asset** field in the Inspector.

3. **Customize Rendering Settings**:
   - Adjust parameters such as:
     - **Splat Size**: Controls the size of splats.
     - **Color Blending**: Defines how overlapping splats blend.

4. **Run the Scene**:
   - Press **Play** to visualize your Gaussian Splats in the Unity scene.

![walking]({{ site.baseurl }}/assets/images/GS1.png){:style="display:block; margin-left:auto; margin-right:auto"}

---

### **Step 4: Experiment with Rendering Options**

1. **Gaussian Falloff**:
   - Tweak the falloff function for splats in the shader or Inspector to achieve desired effects.

2. **Debug Tools**:
   - Use built-in debug options to visualize bounding boxes, normals, or splat points for better understanding and optimization.

   ![walking]({{ site.baseurl }}/assets/images/GS2.png){:style="display:block; margin-left:auto; margin-right:auto"}

3. **Optimize Performance**:
   - **LOD**: Implement Level of Detail (LOD) systems for large datasets.
   - **GPU Instancing**: Enable GPU instancing for improved performance.

---

### **Next Steps**

- **Try Different Data Sets**: Load various splat files to explore their visual potential.
- **Customize Shaders**: Modify the Gaussian Splatting shaders for unique effects.
- **Share Your Work**: Build and deploy your project to share interactive visualizations.

Let us know how Gaussian Splatting transforms your Unity projects! 🌟

### **References**

1. **Unity Plugin for Gaussian Splatting**  
   The official Unity Gaussian Splatting plugin repository by Aras Pranckevičius:  
   [https://github.com/aras-p/UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting)  

2. **3D Gaussian Splatting for Real-Time Radiance Field Rendering**  
   Explore the original implementation and research on Gaussian Splatting:  
   [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)  

Would you like to include additional resources or citations?