---
layout: post
title:  "Designing Dynamic RPG Systems: Quick Solutions for Health and Mana in Unreal Engine"
author: Gurwinder
categories: [ Game Development]
# image: assets/images/present-mon-fps.webp
featured: true
hidden: false
---

* Add two float variables to Third Person Controller Blueprint ‘Health’ and ‘Mana’.

* Create a new UI Widget: In the Content Browser, right-click in the UI folder and select “User Interface” and then “Widget Blueprint.” Name the widget something like “BarWidget.”

![walking]({{ site.baseurl }}/assets/images/unreal-rpg-1.webp)

* Add Two Progress Bar in Designer UI: Open the BarWidget Blueprint and design the health bar using widgets like images and progress bars. Customize the appearance, size, and color of the health bar to fit your game’s aesthetics.

![walking]({{ site.baseurl }}/assets/images/unreal-rpg-2.webp)

* Bind the Health variables to the UI: Bind the Health and Mana variables to the value properties of their respective progress bar widgets. This will update the progress bars based on the current health and mana values.

![walking]({{ site.baseurl }}/assets/images/unreal-rpg-3.webp)

* Then bind Value of Second Bar — Mana to UI

![walking]({{ site.baseurl }}/assets/images/unreal-rpg-4.webp)

* Display the Health and Mana Bars in the game: In the Level Blueprint or the Blueprint of the character or actor you want to show the health and mana bars for, add the BarWidget to the viewport when necessary. Use the “Create Widget” and “Add to Viewport” nodes to achieve this. You can customize changes to variables in Event Graph of Third Controller blueprint such as Health being dependent upon Strength and mana dependent upon Intelligence to be added to viewport.

![walking]({{ site.baseurl }}/assets/images/unreal-rpg-5.webp)

* You can add additional functionality as needed, such as adding visual effects or changing the appearance of the bars based on certain conditions.

![walking]({{ site.baseurl }}/assets/images/unreal-rpg-6.webp)

As always, refer to the Unreal Engine documentation and tutorials for more detailed instructions and examples on working with UI widgets and implementing health and mana bars in Unreal Engine.