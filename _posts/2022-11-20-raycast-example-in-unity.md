---
layout: post
title:  "RAYCAST EXAMPLE IN UNITY"
author: Gurwinder
categories: [ Game Development, Unity ]
image: assets/images/raycast-example-1.gif
featured: false
hidden: false
---
These are the step-by-step example of using basic ray-cast in unity. In the given scene we will ray-cast cube to detect sphere and follow it up.

Create a GameObject: Start by creating a GameObject that will act as the source or origin of your raycast and the object that will move towards the target GameObject.

Attach a script: Attach a script to the source GameObject that will handle the raycasting and movement logic. Right-click on the GameObject, select `“Create Empty”` and attach a C# script to it.
Implement the raycast and movement logic: Open the script in your preferred code editor and implement the raycasting and movement logic. Here’s an example:

```c#
using UnityEngine;

public class RaycastAndMove : MonoBehaviour
{
    public Transform target;  // The target GameObject to move towards
    public float moveSpeed = 5f;  // The movement speed

    void Update()
    {
        // Perform the raycast
        Ray ray = new Ray(transform.position, transform.forward);
        RaycastHit hit;
        
        if (Physics.Raycast(ray, out hit))
        {
            // Check if the raycast hit the target GameObject
            if (hit.collider.gameObject == target.gameObject)
            {
                // Move towards the target GameObject
                Vector3 direction = target.position - transform.position;
                transform.Translate(direction.normalized * moveSpeed * Time.deltaTime);
            }
        }
    }
}
```
Note: In this example, the script performs a raycast from the source GameObject’s position in the forward direction `(transform.position, transform.forward)`. If the raycast hits an object and it matches the target GameObject, it calculates the direction from the source to the target `(Vector3 direction = target.position - transform.position)` and moves towards it `(transform.Translate(direction.normalized * moveSpeed * Time.deltaTime))`.

Assign the target GameObject: In the Unity Editor, assign the target GameObject to the target variable of the script. This can be done by dragging and dropping the target GameObject into the appropriate field in the Inspector window.
Attach the script to the source GameObject: Drag and drop the script onto the source GameObject in the Unity Editor’s Inspector window to attach it.
Test the raycast and movement: Run your scene in the Unity Editor or on a target platform. The source GameObject will perform a raycast and move towards the target GameObject if the raycast hits it.
Note: Make sure the target GameObject has appropriate colliders attached to enable collision detection with the raycast.

## TAG Alternative:

To check if a raycast hits a GameObject with a specific tag in Unity, you can modify the raycast logic and include a tag check. Here’s an example:

```c#
using UnityEngine;

public class RaycastWithTag : MonoBehaviour
{
    public string targetTag = "YourTag"; // The tag to check against
    public float raycastDistance = 100f; // The distance of the raycast

    void Update()
    {
        if (Input.GetMouseButtonDown(0)) // Perform raycast on left mouse button click
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit, raycastDistance))
            {
                // Check if the raycast hit a GameObject with the specified tag
                if (hit.collider.CompareTag(targetTag))
                {
                    // Do something when the raycast hits a GameObject with the specified tag
                    Debug.Log("Raycast hit GameObject with tag: " + targetTag);
                }
            }
        }
    }
}
```

![walking]({{ site.baseurl }}/assets/images/raycast-example-1.gif)

Note: In this example, the script performs a raycast from the main camera’s position to the mouse position `(Camera.main.ScreenPointToRay(Input.mousePosition))`. If the raycast hits an object within the specified raycastDistance, it checks if the hit object has the specified targetTag using `hit.collider.CompareTag(targetTag)`. If the tags match, it executes the desired logic (in this case, logging a message to the console).

Replace `"YourTag"` with the actual tag you want to check against. You can attach this script to a GameObject in your scene to perform the raycast and tag check.

For more information, refer to the Unity documentation on raycasting and tags.