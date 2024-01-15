---
layout: post
title:  "Mastering Frame Rates: Discover the True FPS with PresentMon"
author: Gurwinder
categories: [ Game Development]
image: assets/images/present-mon-fps.webp
featured: true
hidden: false
---

PresentMon is a tool used for capturing frame time data during application runtime, which can then be used to calculate frames per second (FPS). Here’s a general process for using PresentMon to calculate FPS:

1. Download PresentMon: Obtain PresentMon from the official GitHub repository or other trusted sources. Ensure you have the latest version compatible with your system.
2. Launch PresentMon: Open a command prompt or PowerShell window and navigate to the directory where PresentMon is located.
3. Start Capturing Data: Run the following command to start capturing frame time data:
PresentMon64.exe -output filename.csv
4. Replace “filename” with the desired name for the output file. This command instructs PresentMon to capture frame time data and save it to a CSV file.
5. Run Your Application: Launch the application or game for which you want to calculate FPS. Let it run for a reasonable duration to capture enough frames for accurate FPS calculation.
6. Stop Capturing Data: Press Ctrl+C in the command prompt or PowerShell window to stop PresentMon from capturing data. This will terminate the data collection process and finalize the output file.
7. Analyze the CSV File: Open the generated CSV file using a spreadsheet software like Microsoft Excel or Google Sheets. The CSV file contains frame time data, including timestamps and durations for each frame.
8. Calculate FPS: To calculate FPS from the frame time data, use the following formula:
`FPS = 1 / (Average Frame Time)`

Calculate the average frame time by taking the sum of all frame durations and dividing it by the total number of frames captured. Then, compute the FPS value using the formula.

```
Frame 1: Frame Time = 0.0167 seconds (60 FPS)
Frame 2: Frame Time = 0.0200 seconds (50 FPS)
Frame 3: Frame Time = 0.0143 seconds (70 FPS)

Average Frame Time = (0.0167 + 0.0200 + 0.0143) / 3 = 0.0170 seconds

Average FPS = 1 / 0.0170 = 58.82 FPS (approximately)
```

PowerShell script to postprocess CSV dumps in a folder

```powershell
Function PostProcess{
    $logs_dir='C:\PresentMon\Logs'
    $logs_names=Get-ChildItem -Path $logs_dir -Filter '*.csv'

    foreach ($log in $logs_names){
        $proc_name=0
        $proc_id=0
        $total_frames=[double]0
        $minimum_msBetweenPresents=0
        $maximum_msBetweenPresents=0
        $msGPUActive=0
        $msBetweenPresents=0

        $logs_path=$logs_dir+'\'+$log
        
        $csv=import-csv $logs_path
        $proc_name=$csv.Application[0]
        $proc_id=$csv.ProcessID[0]
        $csv | foreach-object {
            $msBetweenPresents+=[double]$_.msBetweenPresents
            $msGPUActive+=[double]$_.msGPUActive
            if ([double]$_.msBetweenPresents -gt $maximum_msBetweenPresents) {$maximum_msBetweenPresents=[double]$_.msBetweenPresents}
            if ($minimum_msBetweenPresents -eq 0) {$minimum_msBetweenPresents=$maximum_msBetweenPresents}
            if ([double]$_.msBetweenPresents -lt $minimum_msBetweenPresents) {$minimum_msBetweenPresents=[double]$_.msBetweenPresents}
            $total_frames++
            }
        $logsData += @([pscustomobject]@{
            Process_Name=$proc_name;
            Process_ID=$proc_id;
            Frames=$total_frames;
            Average_FPS= [math]::Round($($total_frames*1000/$msBetweenPresents),2);
            Min_FPS= [math]::Round($(1000/$maximum_msBetweenPresents),2);
            Max_FPS= [math]::Round($(1000/$minimum_msBetweenPresents),2);
            GPU_Util=$($msGPUActive*100/$msBetweenPresents)
        })
        
        }
        Write-Output $logsData | Format-Table -AutoSize
}
```

By following these steps, you can utilize PresentMon to capture frame time data and calculate FPS for your application or game. The accuracy of the calculated FPS will depend on the duration and number of frames captured, so capturing data over a longer duration with a sufficient number of frames will yield more reliable results.

Image Credits: `Aslysun / Shutterstock`