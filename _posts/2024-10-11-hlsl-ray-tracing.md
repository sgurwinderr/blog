---
layout: post
title:  "HLSL Ray Tracing: Crafting Realistic Scenes in Unity, One Ray at a Time"
author: Gurwinder
categories: [ Game Development, Unity ]
image: assets/images/unity-raytracing.jpg
featured: false
hidden: false
---

# Building a Ray Tracing Shader in Unity with HLSL

Ray tracing is one of the most powerful techniques in computer graphics for achieving realistic lighting and reflections. Unlike traditional rasterization, which processes geometry one triangle at a time, ray tracing simulates how rays of light interact with objects in the scene. In this article, I will guide you through the process of building a simple ray tracing shader using HLSL in Unity. We’ll not only look at code but also dive into the mathematics behind ray-object intersections, lighting, and reflections.

---

## Mathematical Foundations of Ray Tracing

In ray tracing, the core idea is to trace the path of a ray from the camera through the scene. This ray may hit objects, reflect off surfaces, or absorb light. The final color of each pixel is calculated by accumulating light values from these interactions.

### Ray Equation

A ray is defined by an origin point \( O \) and a direction \( D \). For any scalar \( t \geq 0 \), the position of the ray at \( t \) is:

\[
P(t) = O + tD
\]

Where:
- \( O \) is the ray origin (typically the camera position).
- \( D \) is the ray direction (from the camera through the pixel).
- \( t \) is a scalar representing the distance from the origin.

Let’s now break this down into shader code.

---

## 1. Vertex and Fragment Shader Setup

Every HLSL shader begins with a vertex shader and fragment shader. In the context of ray tracing, the vertex shader primarily transforms vertices to screen space, while the fragment shader is where the ray tracing calculations happen.

```hlsl
struct appdata {
    float4 vertex : POSITION;
    float2 uv : TEXCOORD0;
};

struct v2f {
    float2 uv : TEXCOORD0;
    float4 vertex : SV_POSITION;
};

v2f vert(appdata v) {
    v2f o;
    o.vertex = UnityObjectToClipPos(v.vertex); // Transforms vertex
    o.uv = v.uv; // Passes UV coordinates
    return o;
}
```

### Explanation

- The **vertex** function transforms 3D object positions to 2D screen space using `UnityObjectToClipPos`.
- The UV coordinates are passed from the vertex to the fragment shader. These UVs represent the pixel location on the screen where rays will be cast.

---

## 2. The Ray Structure

In ray tracing, rays are key to everything. A ray in 3D space can be represented as a struct with two main components: the origin and the direction.

```hlsl
struct Ray {
    float3 origin;
    float3 dir;
};
```

- **origin**: The starting point of the ray, which is typically the camera's position.
- **dir**: The normalized direction vector in which the ray travels.

The ray equation, \( P(t) = O + tD \), helps us compute points along the ray by adjusting \( t \). For each intersection, the goal is to solve for \( t \), where \( P(t) \) lies on an object’s surface.

---

## 3. Ray-Sphere Intersection

One of the simplest objects to ray trace is a sphere. The equation of a sphere with radius \( r \) and center \( C = (x_c, y_c, z_c) \) is:

\[
(x - x_c)^2 + (y - y_c)^2 + (z - z_c)^2 = r^2
\]

To find out where a ray intersects this sphere, substitute the ray equation into the sphere equation:

\[
(O_x + tD_x - C_x)^2 + (O_y + tD_y - C_y)^2 + (O_z + tD_z - C_z)^2 = r^2
\]

This expands into a quadratic equation in \( t \):

\[
at^2 + bt + c = 0
\]

Where:
- \( a = D \cdot D \)
- \( b = 2 (O - C) \cdot D \)
- \( c = (O - C) \cdot (O - C) - r^2 \)

We solve this quadratic equation to find the possible values of \( t \) (the points where the ray intersects the sphere). If the discriminant \( b^2 - 4ac \) is non-negative, the ray intersects the sphere.

Here’s how that looks in code:

```hlsl
HitInfo RaySphere(Ray ray, float3 sphereCentre, float sphereRadius) {
    HitInfo hitInfo = (HitInfo)0;
    float3 offsetRayOrigin = ray.origin - sphereCentre;
    
    float a = dot(ray.dir, ray.dir);
    float b = 2 * dot(offsetRayOrigin, ray.dir);
    float c = dot(offsetRayOrigin, offsetRayOrigin) - sphereRadius * sphereRadius;
    
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0) {
        float dst = (-b - sqrt(discriminant)) / (2 * a);
        if (dst >= 0) {
            hitInfo.didHit = true;
            hitInfo.dst = dst;
            hitInfo.hitPoint = ray.origin + ray.dir * dst;
            hitInfo.normal = normalize(hitInfo.hitPoint - sphereCentre);
        }
    }
    return hitInfo;
}
```

### Mathematical Explanation

- The quadratic formula \( t = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \) determines the intersection points.
- If \( t \) is positive, the intersection point is in front of the ray origin, meaning a visible hit.
- Once the hit is confirmed, we calculate the normal at the hit point as:

\[
\text{normal} = \frac{P(t) - C}{r}
\]

This normalized vector is essential for calculating lighting and reflections.

---

## 4. Ray-Triangle Intersection Using the Möller-Trumbore Algorithm

Triangles are ubiquitous in 3D graphics. The Möller-Trumbore algorithm is an efficient way to compute ray-triangle intersections. The algorithm uses barycentric coordinates to determine whether a ray intersects the triangle.

Given triangle vertices \( V_0 \), \( V_1 \), and \( V_2 \), and a ray \( P(t) = O + tD \), we compute the intersection using:

1. **Edge vectors**: \( E_1 = V_1 - V_0 \), \( E_2 = V_2 - V_0 \)
2. **Vector cross products**: Using these vectors, we solve for \( t \), \( u \), and \( v \), where \( u \) and \( v \) are barycentric coordinates and must satisfy \( u + v \leq 1 \).

The ray intersects the triangle if the following conditions hold:
- \( t \geq 0 \) (the intersection point is in front of the ray).
- \( 0 \leq u \leq 1 \) and \( 0 \leq v \leq 1 \).

Here’s the code:

```hlsl
HitInfo RayTriangle(Ray ray, Triangle tri) {
    float3 edgeAB = tri.posB - tri.posA;
    float3 edgeAC = tri.posC - tri.posA;
    float3 normalVector = cross(edgeAB, edgeAC);
    float3 ao = ray.origin - tri.posA;
    float3 dao = cross(ao, ray.dir);

    float determinant = -dot(ray.dir, normalVector);
    float invDet = 1 / determinant;
    
    float dst = dot(ao, normalVector) * invDet;
    float u = dot(edgeAC, dao) * invDet;
    float v = -dot(edgeAB, dao) * invDet;
    float w = 1 - u - v;

    HitInfo hitInfo;
    hitInfo.didHit = determinant >= 1E-6 && dst >= 0 && u >= 0 && v >= 0 && w >= 0;
    hitInfo.hitPoint = ray.origin + ray.dir * dst;
    hitInfo.normal = normalize(tri.normalA * w + tri.normalB * u + tri.normalC * v);
    hitInfo.dst = dst;
    return hitInfo;
}
```

### Mathematical Explanation

- The algorithm uses vector math to determine whether the intersection point lies within the triangle.
- \( t \) is the distance along the ray to the hit point.
- \( u \) and \( v \) are the barycentric coordinates, ensuring the intersection lies within the triangle.

---

## 5. Lighting and Reflections

In ray tracing, light is calculated by simulating how rays reflect off surfaces. For each intersection, we need to check how much light reaches the surface and calculate the reflection or refraction of the ray.

The basic lighting model is given by:

\[
L = C + \sum_i (R_i \cdot L_i)
\]

Where:
- \( C \) is the surface's base color.
- \( R_i \) is the reflection coefficient for each light source \( i \).
- \( L_i \) is the incoming light intensity from each source.

We trace the rays through multiple bounces to simulate light reflecting off surfaces:

```hlsl
float3 Trace(Ray ray, inout uint rngState) {
    float3 incomingLight = 0;
    float

3 rayColour = 1; // Initialize to white

    for (int i = 0; i < MAX_BOUNCES; i++) {
        HitInfo hitInfo = Cast(ray); // Cast the ray to find intersections

        if (hitInfo.didHit) {
            float3 normal = hitInfo.normal;
            // Reflect ray based on the surface normal
            ray = ReflectRay(hitInfo.hitPoint, normal, ray);
            rayColour *= hitInfo.color; // Accumulate colors
            incomingLight += CalculateLighting(hitInfo);
        } else {
            break; // No intersection; exit the loop
        }
    }

    return incomingLight * rayColour; // Final color output
}
```

### Explanation

- **Ray Bounces**: For each bounce, we update the ray based on the surface normal. This simulates the light reflecting off surfaces.
- **Color Accumulation**: Each hit accumulates color contributions from the surface and incoming light.
- **Final Color Calculation**: The final output is a product of incoming light and the accumulated color from reflections.

### Reflective Rays

The reflection direction can be calculated using:

\[
R = D - 2(D \cdot N)N
\]

Where:
- \( R \) is the reflected ray.
- \( D \) is the incident ray direction.
- \( N \) is the normal at the hit point.

---

## Conclusion

By understanding both the code and the underlying mathematics, you can create a ray tracing shader in Unity that simulates realistic lighting and reflections. This method is computationally intensive but allows for stunning visual effects that can enhance any 3D project.

The combination of HLSL for shader programming and the mathematical principles of ray tracing offers a powerful toolset for developers looking to push the boundaries of real-time graphics. Experiment with different shapes, lighting conditions, and reflection models to create your unique visual style!

---

Feel free to adjust any sections or add your insights!